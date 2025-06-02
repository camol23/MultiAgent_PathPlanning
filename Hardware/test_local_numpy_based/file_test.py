
# f = open("./logging_data/dist_test0.txt", "x")

with open("./logging_data/dist_test0.txt", "a") as f:
  f.writelines("\n")
  f.writelines("New test")