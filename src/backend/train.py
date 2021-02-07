from backend.data import getVideo,sz_videos, get_videoId, getCaption

def get_model():
	# -_-
	pass    

print(sz_videos())
vId = get_videoId(10)
print(getCaption(vId))
frames = getVideo(vId)
print("Number of frames %d " % (len(frames)))
#cv2.namedWindow( "Display window", cv2.WINDOW_NORMAL )
#for img in frames:
#    cv2.imshow( "Display window", img );
#cv2.destroyAllWindows()
    

