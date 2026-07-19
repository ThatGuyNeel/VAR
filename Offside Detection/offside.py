import cv2
import torch
import numpy as np
import os
import model.sportsfield_release.utils.image_utils as utils
import model.sportsfield_release.utils.warp as warp


def convertPoint3Dto2D(homography: torch.Tensor, p: list[int], w: int, h: int) -> list[float]:
    '''Function that converts a point on the 3D image to a point on the 2D field.
    The function takes as input:
    - The homography tensor;
    - The point to convert;
    - The width of the image of the point to convert;
    - The height of the image of the point to convert.\n
    The function returns as output the x and y of the warped position.'''
    x = torch.tensor(p[0] / w - 0.5).float()
    y = torch.tensor(p[1] / h - 0.5).float()
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy_warped = torch.matmul(homography.cpu(), xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # [-1, 1] -> [0, 1]
    x_warped = (x_warped.item() * 0.5 + 0.5) * 1050
    y_warped = (y_warped.item() * 0.5 + 0.5) * 680

    return [x_warped, y_warped]



def convertPoint2Dto3D(homography: torch.Tensor, p: list[int], w: int, h: int) -> list[float]:
    '''Function that converts a point on the 2D field to a point on the 3D image.
    The function takes as input:
    - The inverted homography tensor;
    - The point to convert;
    - The width of the image onto which to convert the point;
    - The height of the image onto which to convert the point.\n
    The function returns as output the x and y of the warped position.
    '''
    x = torch.tensor(p[0] / 1050 - 0.5).float()
    y = torch.tensor(p[1] / 680 - 0.5).float()
    xy = torch.stack([x, y, torch.ones_like(x)])
    xy_warped = torch.matmul(homography.cpu(), xy)  # H.bmm(xy)
    xy_warped, z_warped = xy_warped.split(2, dim=1)

    # we multiply by 2, since our homographies map to
    # coordinates in the range [-0.5, 0.5] (the ones in our GT datasets)
    xy_warped = 2.0 * xy_warped / (z_warped + 1e-8)
    x_warped, y_warped = torch.unbind(xy_warped, dim=1)
    # [-1, 1] -> [0, 1]
    x_warped = (x_warped.item() * 0.5 + 0.5) * w
    y_warped = (y_warped.item() * 0.5 + 0.5) * h

    return [x_warped, y_warped]


def putPng(image, tag, position) -> None:
    if tag.shape[2] == 4:
    # Separate the RGBA channels
        b, g, r, a = cv2.split(tag)

        # Create a mask and its inverse using the alpha channel
        new_mask = cv2.merge([a, a, a])
        inverse_mask = cv2.bitwise_not(new_mask)
        
        # Define the dimensions of the image to overlay
        height, width = tag.shape[:2]

        # Specify the position (x, y) where you want to insert the overlaid image
        x,y = position[0], position[1]

        # Create the ROI on the background image
        roi = image[y:y+height, x:x+width]
        
        # Use the inverse mask to black out the ROI area in the background
        background_bg = cv2.bitwise_and(roi, roi, mask=inverse_mask[:, :, 0])
        
        # Use the mask to extract the part of the overlaid image
        overlay_fg = cv2.bitwise_and(tag[:, :, :3], tag[:, :, :3], mask=new_mask[:, :, 0])
        
        #  Combine the background and the overlaid image
        combined = cv2.add(background_bg, overlay_fg)
        
        # Insert the combination into the ROI of the background
        image[y:y+height, x:x+width] = combined


def drawOffside(pathImage: str, team: str, colors: dict[str, np.ndarray], homography:torch.Tensor, defender:list[list[int]], attacker: list[list[int]], goalkeeper: list[list[int]]=0) -> int:
    ''' Function that calculates the offside and the players' positions on the 2D and 3D images.
    The function takes as input:
    - The path of the 3D image
    - The homography tensor
    - The list of defenders' positions in the 3D image
    - The list of attackers' positions in the 3D image 
    - The position of the goalkeeper in the 3D image 
    The function returns as output the number of attackers in an offside position and saves the processed image in a result folder'''
    image = cv2.imread(pathImage)
    pitch2D = cv2.imread("model/sportsfield_release/data/world_cup_template.png")
    offside_tag = cv2.imread('GUI/src/images/resizedTag.png',  cv2.IMREAD_UNCHANGED)

    # Calculate width and height of the photo
    w = len(image[0])
    h = len(image)
    side = ''
    offside = []
    attacker2D = []
    defender2D = []

    if team == 'A':
        c_def = colors['Team B'].tolist()
        c_att = colors['Team A'].tolist()
    else:
        c_def = colors['Team A'].tolist()
        c_att = colors['Team B'].tolist()
 

    # Calculate the players' positions in 2D
    for p in attacker:
        p_att = convertPoint3Dto2D(homography, [round((abs(p[0]+p[2])/2)), (p[3])], w, h)
        attacker2D.append(p_att)
        
    for p in defender:
        p_def = convertPoint3Dto2D(homography, [round((abs(p[0]+p[2])/2)), (p[3])], w, h)
        defender2D.append(p_def)
        
    if goalkeeper != 0:
        p_gk = convertPoint3Dto2D(homography, [round((abs(goalkeeper[0][2]+goalkeeper[0][0])/2)), (goalkeeper[0][3])], w, h)
        if p_gk[0] < 1050//2:
            side = 'left'
        else:
            side = 'right'
        if team == 'B':
            cv2.circle(pitch2D, (int(p_gk[0]), int(p_gk[1])), 10, c_def, -1)
        if team == 'A':
            cv2.circle(pitch2D, (int(p_gk[0]), int(p_gk[1])), 10, c_def, -1)
    else:
        c_left, c_right = 0,0
        for p in defender2D:
            if p[0] < 1050//2:
                c_left += 1
            else:
                c_right += 1
        for p in attacker2D:
            if p[0] < 1050//2:
                c_left += 1
            else:
                c_right += 1
        if c_left > c_right:
            side = 'left'
        else:   
            side = 'right'

    # Draw the offside line and calculate the players in an offside position
    if side == 'left':
        last_def = min(defender2D, key=lambda x: x[0])
        cv2.line(pitch2D, (int(last_def[0]), 0), (int(last_def[0]), 680), (0,255,255), 2)
        invexHomo = torch.inverse(homography)
        p1 = convertPoint2Dto3D(invexHomo, [last_def[0], 0], w, h)
        p2 = convertPoint2Dto3D(invexHomo, [last_def[0], 680], w, h)
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,255), 3)

        for i, p in enumerate(attacker2D):
            if p[0] < last_def[0]:
                offside.append(p)
                #CHANGE FONT
                mediax = round(((attacker[i][2]-attacker[i][0])/2)+attacker[i][0])

                putPng(image, offside_tag, [mediax-65,attacker[i][1]-30])


    if side == 'right':
        last_def = max(defender2D, key=lambda x: x[0])
        cv2.line(pitch2D, (int(last_def[0]), 0), (int(last_def[0]), 680), (0,255,255), 2)
        invexHomo = torch.inverse(homography)
        p1 = convertPoint2Dto3D(invexHomo, [last_def[0], 0], w, h)
        p2 = convertPoint2Dto3D(invexHomo, [last_def[0], 680], w, h)
        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,255,255), 3)

        for i, p in enumerate(attacker2D):
            if p[0] > last_def[0]:
                offside.append(p)
                mediax = round(((attacker[i][2]-attacker[i][0])/2)+attacker[i][0])

                putPng(image, offside_tag, [mediax-65,attacker[i][1]-30])


    for p in attacker2D:
        if p in offside:
            cv2.circle(pitch2D, (int(p[0]), int(p[1])), 12, (0,255,255), -1)
        cv2.circle(pitch2D, (int(p[0]), int(p[1])), 10, c_att, -1)
    for p in defender2D:
        cv2.circle(pitch2D, (int(p[0]), int(p[1])), 10, c_def, -1)


  
    '''The number of attackers in an offside position is calculated and the processed images are saved in the result folder.'''

    playerOffside = len(offside)

    os.chdir('result')
    cv2.imwrite('result3D.jpg', image)
    cv2.imwrite('result2D.png', pitch2D)
    os.chdir('..')
    
    return playerOffside
