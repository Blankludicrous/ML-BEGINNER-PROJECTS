import csv
students = [
    ["Amit", 20, 78],
    ["Priya", 19, 85],
    ["Rahul", 21, 67],
    ["Sneha", 20, 92],
    ["Karan", 22, 45]
]
with open("student_details.csv","w+",newline="") as p:
    writer=csv.writer(p)
    writer.writerow(["name","age","marks"])
    writer.writerows(students)