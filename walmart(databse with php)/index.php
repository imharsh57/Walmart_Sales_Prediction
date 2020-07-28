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
<html lang="en" dir="ltr">
  <head>
    <!-- Latest compiled and minified CSS -->
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">

<!-- jQuery library -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>

<!-- Latest compiled JavaScript -->
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    <meta charset="utf-8">
    <title>Demo</title>

    <link rel="stylesheet" href="mystyle.css">
    
  </head>
  <body>
    <h1 style="padding-left: 4%">Welcome to Walmart</h1>
    
      <ul style="padding-left: 4%">
  <li><a class="active" href="{{url_for('view_template')}}">Home</a></li>
  <li><a href="#news">Top_10 dataset</a></li>
  <li><a href="#contact">Graphs</a></li>
  <li><a href="#about">Accuracy of model</a></li>
</ul>



   <div style="padding-left: 10%;padding-top: 2%;padding-right: 10%"> 

    <!--<form action="{{ url_for('form_data') }}" method="post"> -->

    <form id="form" method="post">

      <label for="store">Store Number</label>
    <input type="number" id="store" name="store_nbr" placeholder="Insert Store Number" min="1" max="45" required>
    <span id="errorMsg" style="display:none;">you can give Store_Number 1 to 45 only</span><br>

      <label for="item">Item Number</label>
    <input type="number" id="item" name="item_nbr" placeholder="Insert Item Number" min="1" max="111" required>
    <span id="errorMsg_item" style="display:none;">you can give Item_Number 1 to 111 only</span><br>

    <label for="weather">Weather</label>
    <select id="weather" name="season">
      <option value=fall>fall</option>
      <option value=spring>Spring</option>
      <option value=summer>Summer</option>
      <option value=winter>Winter</option>
    </select>
  
    <input type="submit" id="submit" name="submit1" value="check" onclick="return myFunction()">
   

    </form>
    <h4><b>No of Units :
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
    ?></b></h4>
  </div>


<div class="container-fluid">
  <h2><b>Graphs</b></h2>
  <div class="col-sm-4">
    <img src="images\bar(10unit)for store,item.png">
  </div>
  <div class="col-sm-4">
    <img src="images\bar(10units)for season.png">
  </div>
  <div class="col-sm-4">
    <img src="images\bar(100units)for store,item.png">
  </div>
  
</div>
<div class="container-fluid">
  <div class="col-sm-3"></div>
 <div class="col-sm-6">
    <img src="images\pie(5units) for store,item.png">
  </div>
</div>

<div style="background-color: black">
    <h4 style="text-align: center;color: white">@CopyRights</h4>

</div>


<script type="text/javascript">   //apply the error message for range of store & item number
$("#store" ).keyup(function() {
  if($('#store').val()<0 || $('#store').val()>46 ){
      $('#errorMsg').show();
  }
  else{
    $('#errorMsg').hide();
  }
});


$("#item" ).keyup(function() {
  if($('#item').val()<0 || $('#item').val()>112 ){
      $('#errorMsg_item').show();
  }
  else{
    $('#errorMsg_item').hide();
  }
});


function myFunction(input) {  //error message after click on button
    var y = document.getElementById("item").value;
    var x = document.getElementById("store").value;
    if (x < 0 || x > 45 ) {
        alert("Store_Number should be between 1-45");
        return false;
    }
    if (y < 0 || y > 111 ) {
        alert("Item_Number should be between 1-111");
        return false;
    }
    return true;

    
}



</script>




  </body>
</html>






