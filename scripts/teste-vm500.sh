#!/usr/bin/env bash
# Teste de tempo de resposta do AirLLM na VM 500 (RTX 3060 Ti 8GB).
# Uso:  bash teste-vm500.sh [repo_do_modelo]
# Padrão: Llama 3.3 70B em FP8 (~75GB de download; a divisão por camadas grava
# mais ~75GB, então precisa de ~160GB livres no disco do cache).
#
# Para o Ollama durante o teste (libera a VRAM) e religa no final, mesmo se der erro.
set -euo pipefail

MODELO="${1:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
BASE="${AIRLLM_TESTE_DIR:-$HOME/airllm-teste}"
export HF_HOME="$BASE/hf-cache"
mkdir -p "$BASE"

echo "== Disco livre em $BASE:"; df -h "$BASE" | tail -1
LIVRE_GB=$(df --output=avail -BG "$BASE" | tail -1 | tr -dc 0-9)
if [ "$LIVRE_GB" -lt 160 ]; then
  echo "!! Só ${LIVRE_GB}GB livres; este modelo precisa de ~160GB. Libere espaço ou use AIRLLM_TESTE_DIR=<outro disco>."
  exit 1
fi

# Ambiente Python isolado (não mexe no Python do sistema nem no n8n)
if [ ! -d "$BASE/venv" ]; then
  python3 -m venv "$BASE/venv"
  "$BASE/venv/bin/pip" install -q --upgrade pip
  "$BASE/venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124
  "$BASE/venv/bin/pip" install -q "git+https://github.com/leandro-mendes/airllm---IaReduzida.git#subdirectory=air_llm" \
    compressed-tensors accelerate
fi

# Libera a GPU: para os serviços do Ollama que estiverem ativos e religa no fim
PARADOS=()
for s in ollama ollama-35b; do
  if systemctl is-active --quiet "$s"; then sudo systemctl stop "$s"; PARADOS+=("$s"); fi
done
religar() { for s in "${PARADOS[@]}"; do sudo systemctl start "$s"; done; echo "== Ollama religado: ${PARADOS[*]:-nenhum}"; }
trap religar EXIT
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv

"$BASE/venv/bin/python" -I - "$MODELO" <<'EOF'
import sys, time, torch
from airllm import AutoModel

modelo = sys.argv[1]
t0 = time.time()
model = AutoModel.from_pretrained(modelo, delete_original=True)
print(f"== Download + divisão por camadas: {time.time()-t0:.0f}s")

pergunta = "Explique em 3 frases o que é um firewall pfSense."
ids = model.tokenizer([pergunta], return_tensors="pt", truncation=True, max_length=128).input_ids.cuda()
N = 10
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = model.generate(ids, max_new_tokens=N, use_cache=True, return_dict_in_generate=True)
dt = time.time() - t0
print("== Resposta (primeiros tokens):", model.tokenizer.decode(out.sequences[0][ids.shape[1]:]))
print(f"== {N} tokens em {dt:.0f}s  ->  {dt/N:.1f} s/token")
print(f"== VRAM de pico: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
EOF
