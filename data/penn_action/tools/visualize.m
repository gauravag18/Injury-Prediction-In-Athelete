% visualize the action video sequences with the joint annotation

for i = 1:100:2326
    % read the i_th annotation
    annotation = load(sprintf('labels/%04d',i));
    % create point display of human joints
    vol = CreatePointLightDisplay(annotation);
    % display every 3 frame of the video 
    % and show joint annotation side-by-side
    for j = 1:3:annotation.nframes
        im = imread(sprintf('frames/%04d/%06d.jpg',i,j));
        imshow([im repmat(vol(:,:,j),[1,1,3])]);
        pause(0.1);
    end
end