def round_to_closest_multiple_of(num, c):
    return num + (num - num % c) % c

print(round_to_closest_multiple_of(11.2, 2))