<?php
if (isset($_POST['upload_submit'])) {
    if (!empty($_FILES['file']['tmp_name'])) {
        if (move_uploaded_file($_FILES['file']['tmp_name'], 'uploads/' . $_FILES['file']['name'])) {
            $filename = 'uploads/' . $_FILES['file']['name'];
            $team = $_POST['teams'];

            if ($team == 'team1') {
                $attack = 1;
            } elseif ($team == 'team2') {
                $attack = 2;
            } else {
                echo "<script>alert('Select Team Please!')</script>";
                echo "<script>history.back()</script>";
                return;
            }

            $full_command = "python3 offside.py --image_path " . $filename . " --Attack " . $attack;
            $output = shell_exec($full_command);

            // Generate the filename for the decision file
            $filename_decision = 'output/' . $_FILES['file']['name'];

            // Check if the decision file exists
            if (file_exists($filename_decision)) {
?>

<!DOCTYPE html>
<html>
<head>
    <style>
        /* CSS styles */

        body {
            font-family: Arial, sans-serif;
            background-color: #f2f2f2;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 800px;
            margin: 40px auto;
            background-color: #fff;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
            border-radius: 4px;
            padding: 20px;
        }

        .result {
            display: flex;
            align-items: center;
        }

        .result img {
            max-width: 400px;
        }

        .result-text {
            margin-left: 20px;
        }

        .result-title {
            font-size: 24px;
            margin-top: 0;
        }

        .go-back-button {
            padding: 10px 20px;
            background-color: #4caf50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 16px;
        }
        .output {
            margin-top: 40px;
            padding: 20px;
            background-color: #f2f2f2;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Result</h2>
        <hr>
        <div class="result">
            <img id="image" src="<?php echo htmlspecialchars($filename_decision); ?>">
            <div class="result-text">
                <h3 class="result-title">Image uploaded and decision done.</h3>
                <button class="go-back-button" onclick="history.back()">Go Back</button>
            </div>
        </div>
        <div class="output">
            <h3>Output:</h3>
            <?php echo $output; ?>
        </div>
    </div>
</body>
</html>

<?php
            } else {
                echo "<p>Error</p>";
            }
        }
    }
}
?>
