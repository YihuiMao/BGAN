

file1 = open(' LPIPS.txt', 'r')
Lines = file1.readlines()
float_v=0.0
for line in Lines:
    chunks = line.split(' ')
    print(chunks[1])
    float_v+=float(chunks[1])
print(float_v/200)