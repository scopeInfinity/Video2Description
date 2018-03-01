<?php
$prefix = "/home/gagan.cs14/btp_VideoCaption/";
function navigation($activeId) {
	$options = array('Home' => 'index.php','All IDs' => 'all_ids.php');
	$content = '<nav class="navbar navbar-default">
	  <div class="container-fluid">
	    <div class="navbar-header">
	      <a class="navbar-brand" href="#">Video2Description</a>
	    </div>
	    <ul class="nav navbar-nav">';
	$counter = 0;
	foreach ($options as $title => $link) {
		$active = "";
		if($activeId == $counter)
			$active = 'class="active"';
		$counter+=1;
		$content.="<li $active><a href='$link'>$title</a></li>";
	}
	$content.='
	    </ul>
	  </div>
	</nav>
	';
	return $content;
}
function get_train_ids() {
	global $prefix;
	$command = "python $prefix/VideoDataset/videohandler.py -strain";
	return system($command);
}
function get_test_ids() {
	global $prefix;
	$command = "python $prefix/VideoDataset/videohandler.py -stest";
	return system($command);
}
function get_val_ids() {
	global $prefix;
	$command = "python $prefix/VideoDataset/videohandler.py -sval";
	return system($command);
}
function predict_ids($ids) {
	global $prefix;
	$command = "python parser.py server -pids $ids";
	return system($command);
}
function predict_fnames($fnames) {
	global $prefix;
	$command = "python parser.py server -pfs $fnames";
	return system($command);
}
?>