from vidstab import VidStab, download_ostrich_video


def videoStabilization_off(mode):

    download_ostrich_video('results/ostrich.mp4')
    if mode==1:
    #VideoStabilization by one type of keypoint detector:
        # Using a specific keypoint detector and customizing keypoint parameters
        stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
        stabilizer.stabilize(input_path='results/ostrich.mp4', output_path='results/stable_video_fast.avi')

        # filled in borders
        stabilizer = VidStab(kp_method='ORB')
        stabilizer.stabilize(input_path='results/ostrich.mp4',
                             output_path='results/stable_video_orb.avi')

    if mode==2:
    #L1-Stabilization in matlab
        print('Matlab implementation')