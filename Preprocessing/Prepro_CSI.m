function Prepro_CSI(folder, action_dir, power, sz)

	start_idx = 100;
    end_idx = 100;
	num_sub = 30;

	% sz: #antenna pair * #subcarrier (30)
	if ~exist('sz', 'var')
		sz = 180;
    end
    
    % Interval: micro-second
	if ~exist('interp_interval', 'var')
		interp_interval = 1000;
    end
    
    % sec: recorded seconds
	if ~exist('sec', 'var')
		sec = 5;
	end

    subfolders = dir(folder);
    label = 1;
    name_name = [];
    name_label = [];
    name_cheak = [];
    
    for sf = 3 : length(subfolders)
        
        subfolder = fullfile(folder, subfolders(sf).name);
        files = dir(subfolder);
        files = files(3 : length(files));
        
        for file = files'
            
            people = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
            
            names = strsplit(file.name, '.');
            name = names(1);
            name_name = [name_name; name];
            csi_mat = [];
		    ori_timestamp = [];
		    csi_trace = read_bf_file([subfolder '/' file.name]);
            
            %clear data
            new_csi_trace = {};
            count = 1;
            for idx = 1:size(csi_trace,1)
                csi = csi_trace{idx};
                if isempty(csi)
                    continue;
                end
                csi = csi.csi;
                if size(csi,1)<2 || size(csi,2) < 3
                    continue;
                end
                new_csi_trace{count} = csi_trace{idx};
                count = count + 1;
            end
            csi_trace = new_csi_trace';
            if size(csi_trace, 1) < sz
                continue;
            end
            
            
            for idx = start_idx:size(csi_trace, 1)-end_idx
                csi = csi_trace{idx};
			    csi = csi.csi;
            
			    %csi = get_scaled_csi(csi);

			    ant_1st = find(csi_trace{idx}.perm == 1);
			    ant_2nd = find(csi_trace{idx}.perm == 2);
			    ant_3rd = find(csi_trace{idx}.perm == 3);

	
			    % When there are 2*3 Tx-Rx ants pairs
			    shape = size(csi, 1)*size(csi, 2)*size(csi,3);
                ori_timestamp = [ori_timestamp, csi_trace{idx}.timestamp_low];

			    % Transform into a column vector with subcarriers of the same ant pair together (30*n)*1
			    %csi = reshape(csi(:, 1:2, :), 4, sz / 4);
			    csi = reshape(csi(:, :, :), 6, sz / 6);
                
                %csi = reshape(csi(:, [ant_1st, ant_2nd], :), 4, sz / 4);
			    csi = reshape(csi', sz, 1);
			    csi_mat = [csi_mat, csi];
            end
            
		    ori_timestamp = ori_timestamp - ori_timestamp(1);
            
            % Check the number of duplicate timestamp
		    % and remove all duplicated samples
		    curr_timestamp = ori_timestamp(1, 1);
		    idx = [];
		    cnt = 0;
		    for i = 2:size(ori_timestamp, 2)
		        if ori_timestamp(1,i) == curr_timestamp
					idx = [idx, i];
					cnt = cnt + 1;
                end
				curr_timestamp = ori_timestamp(1,i);
            end
		    disp 'duplicated number:'  
		    disp (cnt)
            csi_mat(:,idx) = [];
            ori_timestamp(:,idx) = [];

		    % Do interpolation to 1000 CSI/s
		    if ~exist(action_dir, 'dir')
			    mkdir(action_dir)
            end
            
            % Based on their data, they will interpoloate the 10000 data to 5000 data.
            % So we have to change the code that will interpolate the data to 5000 even it's original data is less than 5000.
            
            %new_timestamp = interp_interval:interp_interval:max(sec * 10^6, ori_timestamp(end));  
            %new_timestamp = interp_interval:interp_interval:min(sec * 10^6, ori_timestamp(end));
            
            ori_mat = [];
		    for row = 1:size(csi_mat, 1) 
                if power == 1
				    tmp_row = abs(csi_mat(row,:)).^2;
			    elseif power == 0
				    tmp_row = abs(csi_mat(row,:));
                else
				    tmp_row = csi_mat(row,:);
                end

			    % Interpolate to the end of timestamp
                %tmp_row = interp1(ori_timestamp,tmp_row,new_timestamp);
			    ori_mat = [ori_mat; tmp_row];
            end
            
            size(ori_mat)
            
            term = fix((label-1)/20);
            
            if label <= 20
                rank = label;
            else
                rank = label - fix(label/20)*20;
                if rank == 0
                    rank = 20;
                end
            end
            
            name = {[people(term+1),num2str(rank)]};
            name_label = [name_label; name];
            
            save(cell2mat([action_dir '/' name '.mat']), 'ori_mat');    
            
            label = label + 1;
        
        end
    end
    name_cheak = [name_label,name_name];
    save name_cheak.mat name_cheak;
end
    
