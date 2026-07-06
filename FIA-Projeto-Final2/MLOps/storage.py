"""
Camada de I/O abstrata — filesystem local OU data lake MinIO (S3-compatível).

Um único módulo que troca o backend de persistência por flag de ambiente, mantendo
os call-sites dos scripts quase idênticos. O `path` passado às funções é sempre a
**chave lógica** já usada no projeto (ex.: "Dados/Silver/clean_data.parquet",
"Model/model.pkl"). No backend local ela é um caminho de arquivo; no backend MinIO
é a object key dentro de um bucket único.

Seleção do backend (env):
    STORAGE_BACKEND = "local" (default) | "minio"

MinIO (env, usados só quando STORAGE_BACKEND="minio"):
    MINIO_ENDPOINT     ex.: http://minio:9000  (ou http://localhost:9000)
    MINIO_ACCESS_KEY
    MINIO_SECRET_KEY
    MINIO_BUCKET       bucket único do data lake

Obs.: `boto3`/`pyarrow` são libs de infra/deploy (S3 + Parquet). `boto3` só é importado
quando o backend MinIO é efetivamente usado (import lazy), então o modo local continua sem
novas dependências obrigatórias.
"""
import io
import os
import json
import pickle

import pandas as pd

STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local").lower()

# --- Cliente MinIO/S3 (lazy) ---------------------------------------------------
_s3_client = None


def _bucket() -> str:
    bucket = os.environ.get("MINIO_BUCKET")
    if not bucket:
        raise RuntimeError("STORAGE_BACKEND=minio requer a env MINIO_BUCKET.")
    return bucket


def _client():
    """Cria (uma vez) o cliente boto3 apontando para o MinIO a partir das envs."""
    global _s3_client
    if _s3_client is None:
        import boto3  # import lazy: só quando o backend MinIO é usado

        _s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("MINIO_ENDPOINT"),
            aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY"),
        )
    return _s3_client


def _use_minio() -> bool:
    return STORAGE_BACKEND == "minio"


def _get_bytes(path: str) -> bytes:
    """Lê o objeto/arquivo como bytes (backend conforme STORAGE_BACKEND)."""
    if _use_minio():
        obj = _client().get_object(Bucket=_bucket(), Key=path)
        return obj["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def _put_bytes(data: bytes, path: str) -> None:
    """Grava bytes no objeto/arquivo (backend conforme STORAGE_BACKEND)."""
    if _use_minio():
        _client().put_object(Bucket=_bucket(), Key=path, Body=data)
        return
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


# --- API pública: CSV / Parquet / Pickle / JSON --------------------------------
def read_csv(path: str, **kw) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(_get_bytes(path)), **kw)


def write_csv(df: pd.DataFrame, path: str, index: bool = False, **kw) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=index, **kw)
    _put_bytes(buf.getvalue().encode("utf-8"), path)


def read_parquet(path: str, **kw) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_get_bytes(path)), **kw)


def write_parquet(df: pd.DataFrame, path: str, index: bool = False, **kw) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=index, **kw)
    _put_bytes(buf.getvalue(), path)


def read_pickle(path: str):
    return pickle.loads(_get_bytes(path))


def write_pickle(obj, path: str) -> None:
    _put_bytes(pickle.dumps(obj), path)


def read_json(path: str) -> dict:
    return json.loads(_get_bytes(path).decode("utf-8"))


def write_json(obj, path: str, **kw) -> None:
    kw.setdefault("indent", 2)
    kw.setdefault("ensure_ascii", False)
    _put_bytes(json.dumps(obj, **kw).encode("utf-8"), path)


# --- Listagem (usada p/ ler a união das partições Bronze dt=*/ no modo lake) ---
def list_keys(prefix: str) -> list:
    """Lista as object keys sob um prefixo (só MinIO).

    No backend local retorna [] — os consumidores usam `glob` diretamente no filesystem
    (ver DataPipeline/config.bronze_partition_paths). Só o modo lake precisa desta listagem.
    """
    if not _use_minio():
        return []
    client = _client()
    bucket = _bucket()
    keys, token = [], None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        keys.extend(o["Key"] for o in resp.get("Contents", []))
        if resp.get("IsTruncated"):
            token = resp["NextContinuationToken"]
        else:
            break
    return keys


# --- Utilitários de bucket (usados pelo seed/bootstrap) ------------------------
def ensure_bucket() -> None:
    """Cria o bucket do data lake se ainda não existir (idempotente)."""
    if not _use_minio():
        return
    client = _client()
    bucket = _bucket()
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if bucket not in existing:
        client.create_bucket(Bucket=bucket)


def upload_file(local_path: str, key: str | None = None) -> None:
    """Sobe um arquivo do filesystem para o lake (key = caminho lógico se omitido)."""
    if not _use_minio():
        return
    with open(local_path, "rb") as f:
        _put_bytes(f.read(), key or local_path)
