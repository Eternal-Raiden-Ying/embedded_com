import serial
import threading
import time
import sys

PORT = "/dev/ttyUSB0"
BAUD = 9600

ser = serial.Serial(
    port=PORT,
    baudrate=BAUD,
    bytesize=8,
    parity="N",
    stopbits=1,
    timeout=0.1,
    rtscts=False,
    dsrdtr=False,
)

ser.setDTR(False)
ser.setRTS(False)

print(f"[OPEN] {PORT} @ {BAUD}")
time.sleep(3)

ser.reset_input_buffer()
ser.reset_output_buffer()

def rx_loop():
    buffer = b""

    while True:
        try:
            data = ser.read(128)
            if not data:
                continue

            buffer += data

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                text = line.decode(errors="ignore").strip()

                if text:
                    print(f"\n[ARM] {text}")
                    print("> ", end="", flush=True)

        except Exception as e:
            print(f"\n[RX ERROR] {e}")
            break

threading.Thread(target=rx_loop, daemon=True).start()

print("[READY] 输入机械臂命令，例如：")
print("POSE 15 0 8 -45 0 45 800")
print("输入 exit 退出")
print("> ", end="", flush=True)

while True:
    line = sys.stdin.readline()

    if not line:
        break

    line = line.strip()

    if line.lower() in ("exit", "quit"):
        break

    if not line:
        print("> ", end="", flush=True)
        continue

    cmd = line + "\r\n"
    ser.write(cmd.encode("ascii"))
    ser.flush()

    print(f"[TX] {line}")
    print("> ", end="", flush=True)

ser.close()
print("\n[CLOSED]")
