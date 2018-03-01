<?php
require('utils.php');
include('header.php');
echo navigation(0);
?>
<?php
if(isset($_GET['ids'])) {
	?>
	<div class="panel panel-default">
	<div class="panel-heading">Output</div>
	<div class="panel-body">
		<?php echo predict_ids($_GET['ids']); ?>
	</div>
	</div>
	<?php
}
if(isset($_GET['fnames'])) {
	?>
	<div class="panel panel-default">
	<div class="panel-heading">Output</div>
	<div class="panel-body">
		<?php echo predict_fnames($_GET['fnames']); ?>
	</div>
	</div>
	<?php
}
?>
<div class="panel panel-default">
<div class="panel-heading">Predict Using IDs</div>
<div class="panel-body">
	 <form action="#">
	  <div class="form-group">
	    <label for="ids">Enter ID's (space separated)</label>
	    <input type="ids" class="form-control" id="ids" name="ids">
	  </div>
	  <button type="submit" class="btn btn-default">Submit</button>
	</form> 
</div>
</div>

<div class="panel panel-default">
<div class="panel-heading">Predict Using File Names</div>
<div class="panel-body">
	 <form action="#">
	  <div class="form-group">
	    <label for="fnames">Enter Filenames (space separated)</label>
	    <input type="fnames" class="form-control" id="fnames" name="fnames">
	  </div>
	  <button type="submit" class="btn btn-default">Submit</button>
	</form> 
</div>
</div>

<?php
include('footer.php');
?>