import subprocess
import os
import signal
import sys
import time

def kill_on_port(port):
    """
    Forcefully kills any process listening on the specified port.
    """
    try:
        # Use fuser to find and kill processes on the port
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True)
    except Exception:
        pass

def run():
    backend_cmd = f"{sys.executable} -m uvicorn main:app --host 0.0.0.0 --port 8000"
    frontend_cmd = "cd frontend && npm run dev"

    print("\n" + "="*50)
    print("🚀 INITIALIZING VIDEO UNDERSTANDING HUB")
    print("="*50)

    # 1. Port Cleanup
    print("🧹 Cleaning up ports 8000 and 5173...")
    kill_on_port(8000)
    kill_on_port(5173)
    time.sleep(1)

    # 2. Start Backend
    print("📥 [STARTING] Backend Server (FastAPI)...")
    backend_proc = subprocess.Popen(
        backend_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        preexec_fn=os.setsid,
        cwd="backend"
    )

    # 3. Start Frontend
    print("🌐 [STARTING] Frontend Dev Server (Vite)...")
    frontend_proc = subprocess.Popen(
        frontend_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
        preexec_fn=os.setsid # Create a process group
    )

    print("\n" + "-"*50)
    print("✅ BOTH SERVICES ARE NOW ACTIVE")
    print(f"📍 API:      http://localhost:8000")
    print(f"📍 UI:       http://localhost:5173")
    print("-"*50)
    print("💡 LOGS WILL APPEAR BELOW. PRESS CTRL+C TO TERMINATE.")
    print("="*50 + "\n")

    try:
        while True:
            if backend_proc.poll() is not None:
                print("\n⚠️ [ALERT] Backend process terminated unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("\n⚠️ [ALERT] Frontend process terminated unexpectedly.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n" + "="*50)
        print("🛑 [STOPPING] Forcefully terminating all services...")
        print("="*50)
        
        # 1. Kill the process groups
        for proc in [backend_proc, frontend_proc]:
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGKILL) # Aggressive kill
            except Exception:
                pass
        
        # 2. Cleanup ports just in case
        time.sleep(0.5)
        kill_on_port(8000)
        kill_on_port(5173)
        
        # 3. Final pkill for any remaining python/vite processes in this project
        print("🧹 Final sweep of lingering processes...")
        subprocess.run(["pkill", "-f", "main.py"], capture_output=True)
        subprocess.run(["pkill", "-f", "vite"], capture_output=True)
        
        print("\n✨ ALL SERVICES CLEARED. HUB IS OFFLINE.")
        print("="*50 + "\n")

if __name__ == "__main__":
    run()
