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
    for row in data:
        print(f"hours studied {row[0]}",
              f"attendance {row[1]}",
              f"marks {row[2]}")