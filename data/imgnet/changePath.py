input_file = "imgnet_targets.txt"
output_file = "imgnet_targets_ours.txt"

with open(input_file, 'r') as f:
    lines = f.readlines()
# import pdb
# pdb.set_trace()
# print('done')



#
with open(output_file, 'w') as f:
   for line in lines:
        input_list = line.strip().split('/')
        print(input_list)
        if 'real' not in input_list:
           f.write(line)
        else:
            new_line = '/'.join([input_list[0], input_list[1], input_list[3]])+'\n'
            f.write(new_line)
        # import pdb
        # pdb.set_trace()     
#  new_line = '/'.join([input_list[0], input_list[1], input_list[3]])+'\n'
#        f.write(new_line)

        # if len(parts) >= 3:
        #     new_line = f"{parts[0]} {parts[2]}\n"
        #     f.write(new_line)
        # else:
        #     print(f"Skipping invalid line: {line}")

print("Processing complete.")
