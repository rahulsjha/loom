# README example against a local Loom (make server equivalent, LOOM_AUTH_REQUIRED=false, bundled ClickHouse)

```java
LoomClient loom = LoomClient.builder().baseUrl("http://localhost:7777")
    .org("dev").namespace("local").agent("java-demo").build();
System.out.println("set    -> " + loom.set("seat_pref", "Prefers an aisle seat on long flights."));
System.out.println("get    -> " + loom.get("seat_pref").value());
loom.search("which seat should I book?", 5).hits().forEach(h -> System.out.println("search -> " + h.memoryKey()));
System.out.println("delete -> " + loom.delete("seat_pref"));
System.out.println("get after delete -> found=" + loom.get("seat_pref").found());
```

Output:

```
set    -> SetResult[memoryId=13d02962-0418-4b41-b1ff-1cc61355e8ed, version=1795097377996641000, contentSha256=5792954113ec2c61f17a79e8e6b8830564ee831d7ecb1173dc676152ddbb2b78, status=active, pinned=false]
get    -> "Prefers an aisle seat on long flights."
search -> seat_pref
delete -> DeleteResult[tombstoned=true, found=true, version=1795097384063050000]
get after delete -> found=false
```
