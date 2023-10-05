binary_str = "01010100"  # 二进制字符串

# 将二进制字符串转换为整数
decimal_num = int(binary_str, 2)

print(decimal_num)
print(bin(decimal_num))

print(type(bin(decimal_num)))
decimal_num = 85  # 十进制整数

# 将整数转换为字节序列
bytes_obj = decimal_num.to_bytes(1, byteorder='little')

print(bytes_obj)
