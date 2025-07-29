doc1="Apple","Mango","Grapes"
doc2="Mango","Grapes","Orange"
doc3="Orange","Pineapple"
query=input("Enter your query:")
if query in doc1:
  print("Doc1")

if query in doc2:
  print("Doc2")

if query in doc3:
  print("Doc3")
else:
  print("query not found")
