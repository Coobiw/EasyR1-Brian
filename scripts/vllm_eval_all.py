import argparse
import os
import subprocess
import time
import signal
import requests
from tqdm import tqdm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str)
parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save vLLM server logs")
args = parser.parse_args()

model_dir = args.model_dir
model_basename = os.path.basename(model_dir)
log_dir = Path(args.log_dir)
log_dir.mkdir(parents=True, exist_ok=True)

step_interval = 18
start_step = 18
end_step = 90

def wait_for_server(port=8000, timeout=600, check_interval=10):
    """等待 vLLM 服务启动"""
    print(f"Waiting for vLLM server to start on port {port}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                print("vLLM server is ready!")
                return True
        except Exception as e:
            pass
        time.sleep(check_interval)
    raise TimeoutError(f"vLLM server did not start within {timeout} seconds")

def kill_process_group(process):
    """安全地 kill 进程组"""
    try:
        # 发送 SIGTERM 到整个进程组
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        # 等待进程结束，最多等待10秒
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # 如果进程还没结束，强制 kill
        print("Force killing vLLM server with SIGKILL...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait()
        except:
            pass
    except (ProcessLookupError, PermissionError):
        # 进程已经不存在了
        pass

# 记录成功和失败的检查点
success_checkpoints = []
failed_checkpoints = []

for idx in tqdm(range(start_step, end_step + 1, step_interval), desc="Evaluating checkpoints"):
    ckpt_dir = os.path.join(model_dir, f"global_step_{idx}", "actor", "huggingface")
    model_name = f"{model_basename}_step{idx}"
    
    # 检查检查点是否存在
    if not os.path.exists(ckpt_dir):
        print(f"\n⚠️  Checkpoint not found: {ckpt_dir}")
        failed_checkpoints.append((model_name, "checkpoint not found"))
        continue
    
    print(f"\n{'='*80}")
    print(f"Processing checkpoint: {model_name}")
    print(f"Checkpoint path: {ckpt_dir}")
    print(f"{'='*80}\n")
    
    # 设置日志文件
    log_file = log_dir / f"{model_name}_vllm.log"
    
    # 1. 启动 vLLM 服务（后台运行）
    print(f"Starting vLLM server for {model_name}...")
    print(f"Server logs will be saved to: {log_file}")
    
    with open(log_file, "w") as f:
        serve_process = subprocess.Popen(
            ["bash", "scripts/vllm_serve.sh", ckpt_dir, model_name],
            stdout=f,
            stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            preexec_fn=os.setsid  # 创建新的进程组，方便后续 kill
        )
    
    eval_success = False
    
    try:
        # 2. 等待服务启动
        wait_for_server(port=8000, timeout=600)
        
        # 额外等待一下，确保模型加载完成
        print("Waiting additional 30 seconds for model to fully load...")
        time.sleep(30)
        
        # 3. 运行评估
        print(f"\nRunning evaluation for {model_name}...")
        eval_cmd = [
            "python", "eval_scripts/vllm_agiqa3k_eval.py",
            "--model_name", model_name,
            "--model_port", "8000"
        ]
        eval_result = subprocess.run(eval_cmd, check=True)
        print(f"✅ Evaluation completed successfully for {model_name}")
        eval_success = True
        success_checkpoints.append(model_name)
        
    except TimeoutError as e:
        print(f"❌ Timeout error: {e}")
        print(f"   Check vLLM logs at: {log_file}")
        failed_checkpoints.append((model_name, "server startup timeout"))
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with exit code {e.returncode}")
        print(f"   Check vLLM logs at: {log_file}")
        failed_checkpoints.append((model_name, f"eval failed (exit {e.returncode})"))
        
    except Exception as e:
        print(f"❌ Unexpected error during evaluation: {e}")
        print(f"   Check vLLM logs at: {log_file}")
        failed_checkpoints.append((model_name, str(e)))
    
    finally:
        # 4. Kill vLLM 服务进程（包括所有子进程）
        print(f"\nStopping vLLM server for {model_name}...")
        kill_process_group(serve_process)
        print(f"vLLM server stopped for {model_name}")
        
        # 等待端口释放
        print("Waiting for port to be released...")
        time.sleep(10)

# 打印总结
print(f"\n{'='*80}")
print("EVALUATION SUMMARY")
print(f"{'='*80}")
print(f"\n✅ Successfully evaluated: {len(success_checkpoints)} checkpoints")
for name in success_checkpoints:
    print(f"   - {name}")

if failed_checkpoints:
    print(f"\n❌ Failed: {len(failed_checkpoints)} checkpoints")
    for name, reason in failed_checkpoints:
        print(f"   - {name}: {reason}")
else:
    print(f"\n🎉 All checkpoints evaluated successfully!")

print(f"\n{'='*80}")
print(f"Logs saved to: {log_dir}")
print(f"{'='*80}")