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
with open("student_details.csv","r") as p:
    data=csv.reader(p)
    next(data)
    for row in data:
        name=row[0]
        age=int(row[1])
        marks=int(row[2]) 
        print(f"name {name}"
              f"age {age}"
              f"marks {marks}")