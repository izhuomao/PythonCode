import serial

try:
    ser = serial.Serial('COM3', 115200, timeout=1)
    with open('data/close.csv', 'a') as f: # 建议用 'a' (追加) 模式，防止覆盖
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                f.write(line + '\n')
                f.flush() # 强制刷新写入硬盘，防止断电数据丢失
                print(line)
except serial.SerialException as e:
    print(f"串口错误: {e}")
except KeyboardInterrupt:
    print("程序已停止")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("串口已关闭")
