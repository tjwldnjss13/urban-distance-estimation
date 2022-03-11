function [ x, y, z ] = image_to_world(u, v, d, seqDir)

    % set default
    x = 1000;
    y = 1000;
    z = 1000;

    % check disparity
    if( d == 0 )
        return;
    end

    % get camera parameters
    [ f, sx, sy, u0, v0, b, cX, cY, cZ, tilt ] = get_stereo_cam_params(seqDir);

    if( f == 0 )
        fprintf('error: please provide a valid sequence dir to obtain the correct camera parameters\n');
        return;
    end
    
    fx = f / sx;
    fy = f / sy;

    % compute 3D point in camera system
    zCam = (fx * b) / d;
    xCam = (zCam / fx) * (u - u0);
    yCam = (zCam / fy) * (v0 - v);

    % correct with camera position and tilt angle
    xWorld = xCam + cX;
    zWorld =   zCam*cos(tilt) + yCam*sin(tilt) + cZ;
    yWorld = - zCam*sin(tilt) + yCam*cos(tilt) + cY;

    
    % copy to result
    x = xWorld;
    y = yWorld;
    z = zWorld;

end