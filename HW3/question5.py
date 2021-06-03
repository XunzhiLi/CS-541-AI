def myAtoi(s: str) -> int:
    res = ''
    a = 1
    for i in s:
        if i == ' ':
            continue
        if (len(res) == 0):
            if i == '+':
                a = 1
            if i == '-':
                a = -1
            if i == 0:
                continue
        if (len(res) != 0):
            if a * int(res) < -2147483648:
                return -2147483648

            if a * int(res) > 2147483647:
                return 2147483647
        if(ord(i)>=48 and ord(i) <=57):

            res += i
    res = int(res)
    return a * res

print(myAtoi(" ##2  -12"))