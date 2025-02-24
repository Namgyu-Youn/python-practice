### Relation Algebra
- mode set : “.mode csv”, “.mode column“, “.header on”,
- store file : “.once (temp.txt)”
- read file : “.import temp.txt (Table)”, “.tables”, “.schema”
- select distinct (Row) from (Table) : select set data (not-multiple)
- drop table (Table);
- rename table : alter table (Table1) rename to (Table2);

### Logical Operator
- Set : u(union), n(intersection), -(difference), *(cartesian product)
- Relation : σ(select), π(project), join, division

### When union(u) two relations, numbers of attribute should be same
- Cartesian product : Make all able possible-case
- Logical operator used in SELECT
- comparison : >, >=, <, <=, <>(not same)
- logic : n(and), u(or), ㄱ(not smaller than=’>=’)

### SELECT : exact following attribute
- SELECT ‘거주지=대전’ from ‘회원’
- SELECT ‘회원등급=A and 취미=등산’ from 회원

### PROJECT : exact following Tuple
- PROJECT ‘거주지’ from ‘회원 = ㅠ(‘거주지=회원’)(회원)
- PROJECT ‘이름 and 회원등급’ from ‘회원’ = ㅠ(‘이름 and 회원등급’)(회원)

### JOIN : combine informations from at least two relation
- case : I/O(Inner/Outer)
- JOIN ‘회원 and 대출’

### Division : ‘A(data) / B’ = ‘If B_data is in A_data, return A without B_data’
- Don’t return same attribute (https://chartworld.tistory.com/12)

### Application of relation logics
- SELECT ‘거주지=대전’ PROJECT from ‘이름 and 취미’
- False : o(‘거주지=대전’)ㅠ(‘이름 and 취미’)(회원)
- True : ㅠ(‘이름 and 취미’ (o(‘거주지=대전’))) (회원)
- ㅠ(‘이름=홍길동’ (o(도서)) (회원)

### When SELECT and PROJECT are used at the same time, always use SELECT first.
Practice
- ㅠ(‘loc=Boston’ (o(‘부서명’ from dept)) from emp
- ㅠ(‘담당 업무=Analyst’ (o(‘이름, 입사일자) from emp)) from emp
- ㅠ(‘부서번호=10 (o(‘이름 and 입사일자’))) from emp

---

## SQL
### Basic SQL methods
- DDL(Data Definition Language) : CREATE, ALTER, DROP
- DML(Data Manipulation Language)
- INSERT : add new data or tuple to table
- UPDATE : revise stored data
- DELETE : delete data or tuple stored in table
- SELECT

###  DCL(Data Control Language) : GRANT/REVOKE (give/withdraw acces/use authority)
- ‘;’ should be followed when SELECT ends.
- We need to know type of column, so copying table’s info is comfortable.


### SELECT
- basic structure : SELECT columns FROM table_name [WHERE condition(조건)] ;
- set “SELECT / FROM / WHERE ;” and build it.
- DISTINCT : remove repeated data
- * : all columns

### WHERE : data condition is followed
- ‘comparison operator(>,=,<>)’, ‘AND’, ‘OR’, ‘NOT’ can be used
- String, Date : ‘SalesMan’, ‘1995-02-21’
- multiple value comparison is available using ‘AND, OR’
- False : 5<x<7 / True : 5<x and x<7
- NULL + number = NULL
- IN/NOT IN : same function like python
- LIKE : check if each column include the strings
- % : used with LIKE. check the numbers of string.
- _ : used with LIKE. check the string number is one.
- internal function : COUNT(), MAX/MIN(), AVG(), SUM()

### Keywords which locate in SELECT’s tail
- ORDER BY : sorting keyword. Ascending model is basic.
- GROUP BY : Group data based on column. Following is only available case
- column used in SELECT
- Aggregate function (like COUNT, MAX, MIN)
- HAVING : filter ‘GROUP BY result’
