<?php
require('utils.php');
include('header.php');
echo navigation(1);
?>

<div class="panel panel-default">
<div class="panel-heading">Train ID's</div>
<div class="panel-body"><?php echo get_train_ids(); ?></div>
</div>
<div class="panel panel-default">
<div class="panel-heading">Test ID's</div>
<div class="panel-body"><?php echo get_test_ids(); ?></div>
</div>
<div class="panel panel-default">
<div class="panel-heading">Validation ID's</div>
<div class="panel-body"><?php echo get_val_ids(); ?></div>
</div>

<?php
include('footer.php');
?>