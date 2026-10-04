"""Diagnostic raster of measured lane history; never used for control.

Rows point forward (+base x); columns point right (-base y).
History is stored in odom and rendered in base coordinates at camera time.
This is not SLAM: calibration and odometry errors remain visible as ghosting.
"""
import cv2
import numpy as np
from .lane_history import rotation

SIZE = 480
METRES_PER_PIXEL = .005


def map_pixels(points):
    points = np.asarray(points, float)
    return np.rint(np.column_stack((SIZE/2-points[:,1]/METRES_PER_PIXEL,
                                   SIZE/2-points[:,0]/METRES_PER_PIXEL))).astype(np.int32)


def visible_curves(frames, stamp, pose, max_age_s=10.):
    if pose is None or stamp is None or not np.isfinite(pose).all():
        return []
    return [(side,(curve-pose[:2])@rotation(pose[2]),(stamp-time)/1e9)
            for time,curves in frames if 0 <= stamp-time <= max_age_s*1e9
            for side,curve in curves.items()]


def render_local_map(frames, stamp, pose, target=None, corner=None, archive=None):
    canvas = np.full((SIZE,SIZE,3),20,np.uint8)
    for row in range(0,SIZE,20):
        cv2.line(canvas,(0,row),(SIZE-1,row),(45,45,45),1)
    for col in range(0,SIZE,20):
        cv2.line(canvas,(col,0),(col,SIZE-1),(45,45,45),1)
    for x in (-1.,-.8,-.6,-.4,-.2,0.,.2,.4,.6,.8,1.):
        point=map_pixels([[x,0]])[0]
        cv2.putText(canvas,f'{x:.1f}m',(3,int(point[1])-3),cv2.FONT_HERSHEY_SIMPLEX,.35,(160,160,160),1)
    cv2.line(canvas,(240,0),(240,SIZE-1),(75,75,75),1)
    if pose is not None and np.isfinite(pose).all() and archive:
        for side, world in archive.items():
            points = (np.asarray(world).reshape(-1, 2)-pose[:2])@rotation(pose[2])
            points = points[np.all(np.abs(points) <= 1., axis=1)]
            pixels = map_pixels(points)
            if len(pixels):
                # Point cloud: never connect unrelated old fragments into a
                # fictional lane. Old observations remain visible behind us.
                canvas[pixels[:,1],pixels[:,0]] = (30,110,30) if side == 'left' else (0,75,110)
    for side,curve,age in visible_curves(frames,stamp,pose):
        color = np.array((60,255,60) if side=='left' else (0,165,255))
        color = tuple(int(v) for v in color*max(.2,1-age/10.))
        cv2.polylines(canvas,[map_pixels(curve)],False,color,1)
    # Overlays are current-frame diagnostics, never inserted into history.
    if pose is not None and target:
        path=np.asarray(target.get('center_path',[]),float)
        if path.ndim==2 and path.shape[1]==2 and len(path)>1 and np.isfinite(path).all():
            cv2.polylines(canvas,[map_pixels(path)],False,(255,255,0),2)
        if np.isfinite([target['x_m'],target['y_m']]).all():
            cv2.circle(canvas,tuple(map_pixels([[target['x_m'],target['y_m']]])[0]),5,(255,0,255),-1)
    corner=corner or {}
    if pose is not None and corner.get('pivot_world') is not None:
        pivot=(np.asarray(corner['pivot_world'])-pose[:2])@rotation(pose[2])
        cv2.drawMarker(canvas,tuple(map_pixels([pivot])[0]),(0,0,255),cv2.MARKER_CROSS,14,2)
    robot=map_pixels([[0,0]])[0]
    cv2.circle(canvas,tuple(robot),7,(255,255,255),2)
    cv2.arrowedLine(canvas,tuple(robot),(int(robot[0]),int(robot[1])-25),(255,255,255),2)
    lines=['LOCAL LANE MAP | display only, NOT SLAM',
           'grid 10cm | 1px=5mm | front/rear +/-1m',
           'LEFT green | RIGHT orange | old = dim',
           f"{corner.get('mode','UNKNOWN')} | calibration/odom unverified"]
    if archive:
        lines.append(f'ARCHIVED measured points: {sum(len(p) for p in archive.values())}')
    if pose is None:
        lines.append('NO FRESH CAPTURE-TIME ODOM: map hidden')
    elif not visible_curves(frames,stamp,pose):
        lines.append('NO RECENT LINE: archived points retained')
    for i,line in enumerate(lines):
        cv2.putText(canvas,line,(6,15+15*i),cv2.FONT_HERSHEY_SIMPLEX,.36,(220,220,220),1)
    return canvas
