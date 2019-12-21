%%provided by idealboy, https://github.com/idealboy/Meta-SR-Pytorch/edit/master/prepare_dataset/generate_LR_metasr_X1_X4_mt.m
function generate_LR_metasr_X1_X6()
%% settings
path_save = fullfile('D:', 'Datasets', 'SR', 'Video', 'REDS');
path_src = fullfile('D:', 'Datasets', 'SR', 'Video', 'REDS', 'train_sharp');
% sub_folder = {'000', '011', '015', '020'};
sub_folder = {'269',};
% sub_folder = {};
% for i = 1 : 270
%     sub_folder = cat(1, sub_folder, sprintf('%03d', i - 1));
% end

ext               =  {'*.png'};
img_content        =  [];
for i = 1 : length(ext)
    for j = 1 : length(sub_folder)
        tar_dir = fullfile(path_src, sub_folder{j});
        tmp = dir(fullfile(tar_dir, ext{i}));
        for k = 1 : length(tmp)
            tmp(k).sub_folder = sub_folder{j};
            tmp(k).folder = tar_dir;
        end
        img_content = cat(1,img_content, tmp);
    end
end
num_imgs = length(img_content);

% for n = 1:num_imgs
%     fprintf('Read HR : %3d|%3d  \n', n, num_imgs);
%     tmp = img_content(n);
%     ImHR = imread(fullfile(tmp.folder, tmp.name));
%     img_content(n).ImHR = ImHR;
% %     [~,sub_folder] = fileparts(tmp.folder);
% %     img_content(n).sub_folder = sub_folder;
% end

LR_dir = fullfile(path_save,'LR_bicubic');

if ~exist(LR_dir, 'dir')
    mkdir(LR_dir)
end
    
%% generate and save LR via imresize() with Bicubic

scales = 1.0:0.1:6.0;
steps = step_for_scales(scales);

scale_num = length(scales);

%% look at the original source code, when applying parfor on outter loop, the large variable 'DIV2K_HR' wiil be
%% passed to each workers, and it will be very very slow just for start up the parfor.

%% we just exchange the original outter-loop and inner-loop,
%% so that, when applying parfor on current inner-loop('parfor i = 2:1:scale_num'),
%% it will not cause a heavy load to each workers when passing the shared variable,(image = ImHR;)
%% and, now, for each image, all the resized images (with diffrent scale) will be processed parallelly(speed up)

for k = 1 : num_imgs
%     fprintf('IdxIm=%d\n', k);
    fprintf('Read HR : %3d|%3d  \n', k, num_imgs);
    tmp = img_content(k);
    ImHR = imread(fullfile(tmp.folder, tmp.name));
%     ImHR = img_content(k).ImHR;
%     tar_dir = fullfile(LR_dir, img_content(k).sub_folder);
%     if ~exist(tar_dir, 'dir')
%         mkdir(tar_dir);
%     end
    
    parfor i = 2:1:scale_num
%     for i = 2:1:scale_num
        image = ImHR;
        scale = scales(i);
        step = steps(i);
%         FolderLR = fullfile(tar_dir, sprintf('X%.2f',scale));
        FolderLR = fullfile(LR_dir, sprintf('X%.2f', scale), img_content(k).sub_folder);
        
        if ~exist(FolderLR, 'dir')
            mkdir(FolderLR);
        end

        [h, w, ~]=size(image);
        image =image(1:new_range(h, scale, step),1:new_range(w, scale, step),:);
%         oh = floor(h/scale);
%         ow = floor(w/scale);
        fileName = img_content(k).name;
        NameLR = fullfile(FolderLR, fileName);
        if ~exist(NameLR, 'file')
            image= imresize(image, 1/scale, 'bicubic');
%             image= imresize(image, [oh, ow], 'bicubic');
            imwrite(image, NameLR, 'png');
        end
    end
end
end


%%
function steps = step_for_scales(scales)
num = length(scales);
steps = zeros(1, num);
for i = 1 : num
    if scales(i) == floor(scales(i))
        steps(i) = 1;
    elseif floor(scales(i) * 2) == scales(i) * 2
        steps(i) = 2;
    elseif floor(scales(i) * 5) == scales(i) * 5
        steps(i) = 5;
    else
        steps(i) = 10;
    end
end
end

function out = new_range(len, scale, step)
tmp = scale * step;
out = floor(floor(len / tmp) * tmp);
end