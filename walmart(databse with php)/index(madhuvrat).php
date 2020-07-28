<?php
  $servername = "localhost";
  $username = "root";
  $pass = "";
  $dbname="train";
  $conn = new mysqli($servername, $username, $pass, $dbname);
  if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
  }
?>
<!DOCTYPE html>
<html>
<head>
	<title>Walmart</title>
</head>
<body>
<center>
	<form method="post" action="#">
		Enter Store Number.<input type="text" name="store_nbr"><br><br>
		Enter Item Number.<input type="text" name="item_nbr"><br><br>
		Enter Season.         <input type="text" name="season"><br><br>
		<input type="submit" name="submit1" value="Check">
	</form><br>
	OUTPUT:-
		<?php
		if(isset($_POST['submit1']))
		{
			$store=$_POST['store_nbr'];
			$item=$_POST['item_nbr'];
			$season=$_POST['season'];
			$query = "SELECT * FROM `walmart` WHERE store_nbr='$store' and item_nbr='$item' and season='$season'";
			$Result = mysqli_query($conn, $query);
			if(mysqli_num_rows($Result)==0)
				{
					echo "0";
				}
			while ($row = $Result->fetch_assoc()) {
                echo $row['units']."<br>";
             }
		}
		?>
</center>
</body>
</html>