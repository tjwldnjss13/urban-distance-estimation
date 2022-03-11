function [ f, sx, sy, u0, v0, b, cX, cY, cZ, tilt ] = get_stereo_cam_params(seqDir)

    % default for invalid sequence dir
    f    = 0; % focal length [pixel]
    sx   = 0; % pixel width X [no unit]
    sy   = 0; % pixel width Y [no unit]
    u0   = 0; % principal point X [pixel]
    v0   = 0; % principal point Y [pixel]
    b    = 0; % baseline [meter]
    cX   = 0; % camera lateral position [meter]
    cY   = 0; % camera height above ground [meter]
    cZ   = 0; % camera offset Z [meter]
    tilt = 0; % camera tilt angle [rad] (NOTE: should be adjusted by online tilt angle estimation)

    switch( seqDir )

        % camera setup 1
        case { 'train_1', 'train_2', 'test_1', 'test_2'}
            f    = 1280.097351;
            sx   = 1.0;
            sy   = 1.002822;
            u0   = 512.788788;
            v0   = 170.038092;
            b    = 0.26;
            cX   = 0.0;
            cY   = 1.2;
            cZ   = 1.7;
            tilt = 0.072;
            
        % camera setup 2
        case 'train_3'
            f    = 1255.769043;
            sx   = 1.0;
            sy   = 1.071844;
            u0   = 525.521667;
            v0   = 225.899994;
            b    = 0.234021;
            cX   = 0.0;
            cY   = 1.2;
            cZ   = 1.7;
            tilt = 0.04;
    end
    
end
