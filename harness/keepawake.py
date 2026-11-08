"""Hold ES_SYSTEM_REQUIRED while any experiment process (jobs.py / kl.py) is alive. Changes no settings."""
import ctypes, subprocess, time
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
k = ctypes.windll.kernel32
def alive():
    out = subprocess.run(["powershell", "-NoProfile", "-c",
        "Get-CimInstance Win32_Process | ? { $_.CommandLine -match 'jobs.py|kl.py' -and $_.Name -match 'py' } | measure | % Count"],
        capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")
k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
time.sleep(120)  # let a queue restart finish before the first check
while alive():
    k.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    time.sleep(60)
k.SetThreadExecutionState(ES_CONTINUOUS)
print("experiments finished; released")
