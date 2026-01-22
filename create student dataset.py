import csv 
data=[[1,50,30],
[2,55,35],
[3,60,40],
[4,65,50],
[5,70,55],
[6,75,65],
[7,80,70],
[8,85,80],
[9,90,85],
[10,95,92]]
with open("students_marks.csv","w+",newline="") as s:
    writer=csv.writer(s)
    writer.writerow(["hours_studied","attendance","marks"])
    writer.writerows(data)
with open("students_marks.csv","r",newline="") as s:
    data=csv.reader(s)
    next(data)
    for row in data:
        hours_studied=int(row[0])
        attendance=int(row[1])
        marks=int(row[2])
        print(
        f"hours_studied{hours_studied}",
        f"attendance{attendance}",
        f"marks {marks}"
        )