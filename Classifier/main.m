function results = main(folder, parameters)

%try

files = dir([folder]);
file = files(3);
name = file.name;
label = importdata([folder '/' file.name]);
dataset = label.data(:,end);

for idx = 3:length(files)
    file = files(idx);
    name = file.name;
    data = importdata([folder '/' file.name]);
    data = data.data;
    data = data(:,1:end-1);
    dataset = [dataset, data];
end

dataset = [dataset(:,2:end),dataset(:,1)];

% Set demo dataset for testing code
%dataset = dataset(1:100, :);
    
warning('off', 'all');
[ndata, nfeats] = size(dataset);
nfeats = nfeats - 1; % last column is labels

rmut = parameters.geneticmutation;
rsel = parameters.geneticselection;

parameters.ndata = ndata; parameters.nfeats = nfeats;
maxdata = min(parameters.maxdata, ndata);
maxfeats = ceil(parameters.maxfeatures * nfeats);

% initialize population of chromosomes
for x = 1:parameters.chromosomes
    a01 = ceil(1 * maxdata); % total number of instances used for training
    a01 = max(10, a01); % absolute minimum is ten patterns
    
    a02 = randperm(maxdata); % order of instances
    for y = 1:nfeats
        a03(y) = round(rand);
    end
    
    while sum(a03) > maxfeats
        rand1 = ceil(rand * nfeats);
        a03(rand1) = 0;
    end
    
    % ARTMAP parameters
    a04(1) = rand; % alpha: [0, 0.1]
    a04(2) = rand; % beta: [0.5, 1.0]
    a04(3) = rand; %?
    a04(4) = rand; % epsilon: [0, 0.1]
    a04(5) = rand; % vigilance: [0, 0.5];
    
    population(x, :) = [{a02} {a03} {a04}];
    %population(x, :) = [{a01} {a02} {a03} {a04}];
end
results.initpop = population;
results.totalsearches = 0;
% start GA optimization
iter = 1;
convergence = 0;
results.classifierconvergence = 0;
fit1(1:parameters.chromosomes, 1:2) = 0; % column1 is accuracy, column2 is subpopulation index
time1 = tic;
preds(1:parameters.chromosomes, 1:ndata) = NaN;
while iter <= parameters.geneticgenerations
    clear predsbuffer
    
    for x = 1:parameters.chromosomes
    %parfor (x = 1:parameters.chromosomes, parameters.parallelcores)    
        [fit1(x, 1), predsbuffer{x}] = fam11_Yuna(dataset, population(x, :), parameters);
    end
    
    for x = 1:parameters.chromosomes
        if isempty(predsbuffer{x})
            fit1(x, 1) = 0.001; % if nothing to train, set minimum, nonzero score
        elseif ~isempty(predsbuffer{x})
            %a01 = double(population{x, 1});
            a02 = population{x, 1};
            a03 = predsbuffer{x}; % first column is true class, second column is predictions
            % need to reorder
            % predsbuffer is in the order of a02, from 1:a01
            preds(x, a02(1:a01)) = a03(1:a01, 2);
        end
    end
    results.fit1(:, iter) = fit1(:, 1);
    
    % assign into subpopulations
    %fit1 = alex71(fit1, parameters.HGA);
    
    [fit1, population, preds] = alex71(fit1, population, preds, parameters.HGA);
    
    % test for convergence
    fit2 = sort(fit1(:, 1), 'descend');
    nsel = round(parameters.chromosomes / parameters.HGA);
    %nsel = ceil(parameters.chromosomes * parameters.geneticselection);
    if iter == 1
        converge1 = [mean(fit2(1:nsel, 1)) max(fit2(1:nsel, 1))];
    elseif (iter > 1) && (mean(fit2(1:nsel, 1)) > converge1(1))
        converge1(1) = mean(fit2(1:nsel, 1));
        convergence = 0;
    elseif (iter > 1) && (max(fit2(1:nsel, 1)) > converge1(2))
        converge1(2) = max(fit2(1:nsel, 1));
        convergence = 0;
    else
        convergence = convergence + 1;
    end
    
    disp(iter); iter = iter + 1; 
    % genetic selection, reproduction, and mutation
    % slow escalation
    %  start with selection=0.5, mutation=0.5
    %  at every convergence, selection=selection+0.1, mutation=mutation-0.1
    %  this is to maximize search early, and slowly converge
    %  endpoint at selection=0.9, mutation=0.1
    if (convergence >= parameters.genetictermination) && (parameters.geneticselection == 0.9)
        disp(['FAM optimization convergence at ', num2str(iter)]);
        results.classifiertime = toc(time1);
        results.classifierconvergence = iter;
        iter = parameters.geneticgenerations +1;
    elseif (convergence >= parameters.genetictermination) && (parameters.geneticselection < 0.9)
        parameters.geneticselection = min(0.9, parameters.geneticselection+0.1);
        parameters.geneticmutation = max(0.1, parameters.geneticmutation-0.1);
        convergence = 0;
        disp(['Generation: ', num2str(iter), '; Selection: ', num2str(parameters.geneticselection), '; Mutation: ', num2str(parameters.geneticmutation)]);
        [fit1, population, preds] = subfunc01(fit1, population, preds, parameters);
        results.totalsearches = results.totalsearches + round(parameters.chromosomes * (1 - parameters.geneticselection));
    elseif (convergence < parameters.genetictermination) && (iter <= parameters.geneticgenerations)
        [fit1, population, preds] = subfunc01(fit1, population, preds, parameters);
        results.totalsearches = results.totalsearches + round(parameters.chromosomes * (1 - parameters.geneticselection));
    end
end

if results.classifierconvergence == 0
    results.classifierconvergence = parameters.geneticgenerations;
    results.classifiertime = toc(time1);
end

% end classifier optimization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% begin ensemble optimization

parameters.geneticmutation = rmut;
parameters.geneticselection = rsel;
nchrom = parameters.chromosomes;
% generate initial population
for x = 1:nchrom
    chrom = round(rand(1, nchrom));
    ensemblepop(x, :) = [rand*2 chrom]; % first gene is the lambda for negcorr
    fit2(x, 1) = 0;
end

gencount = 1;
converge1(1:2) = 0;
convergecount = 0;
time1 = tic;
while gencount <= parameters.geneticgenerations
%     for x = 1:nchrom
%     %parfor (x = 1:nchrom, parameters.parallelcores)
%         if fit2(x, 1) == 0
%             [fit2(x, 1), newchrom] = subfunc04(ensemblepop(x, :), preds, dataset(:, nfeats+1));
%             fit2(x, 1) = fit2(x, 1) / (1 + (count(newchrom) / 100));
%             ensemblepop(x, :) = (ensemblepop(x, :) + newchrom) / 2;
%         end
%     end
    
    % 
    for x = 1:nchrom
    %parfor (x = 1:nchrom, parameters.parallelcores)
        if fit2(x, 1) == 0
            a01 = find(ensemblepop(x, 2:end) > 0);
            a02 = preds(a01, :);
            a03 = fit1(a01, 1);
            labels = dataset(:, end);
            
            [preds1, ~, acc, ~] = alexfastvote(a02', labels, a03);
            
            i = 1; bestacc = acc; %bestsize = size(a02, 2);
            while i < size(a02, 1)
                testchromosome = a01;
                testchromosome(:, i) = [];
                testpreds = a02;
                testpreds(i, :) = [];
                testacc = a03;
                testacc(i, :) = [];
                
                    [preds1, ~, acc, ~] = alexfastvote(testpreds', labels, testacc);
                if acc >= bestacc
                    a01 = testchromosome;
                    a02 = testpreds;
                    a03 = testacc;
                    bestacc = acc;
                    i = 1;
                else
                    i = i + 1;
                end
            end
            
            preds2(x, :) = preds1;
            fit2(x, 1) = acc;
            a00 = zeros(1, length(ensemblepop(x, 2:end))); 
            a00(a01) = 1;
            ensemblepop(x, 2:end) = a00;
            
        end
    end
    
    % test for convergence
    fit3 = sort(fit2, 'descend');
    nsel = ceil(parameters.chromosomes * parameters.geneticselection);
    if gencount == 1
        converge1 = [mean(fit3(1:nsel, 1)) max(fit3(1:nsel, 1))];
    elseif (gencount > 1) && (mean(fit3(1:nsel, 1)) > converge1(1))
        converge1(1) = mean(fit3(1:nsel, 1));
        convergecount = 0;
    elseif (gencount > 1) && (max(fit3(1:nsel, 1)) > converge1(2))
        converge1(2) = max(fit3(1:nsel, 1));
        convergecount = 0;
    else
        convergecount = convergecount + 1;
    end
    
    results.fit2(:, gencount) = fit2(:, 1); 
    % genetic selection, reproduction, and mutation
    if (convergecount >= parameters.genetictermination) && (parameters.geneticselection >= 0.9)% && (parameters.geneticmutation >= 0.9)
        disp(['Ensemble optimization convergence at ', num2str(gencount)]);
        results.ensembletime = toc(time1);
        results.ensembleconvergence = gencount;
        gencount = parameters.geneticgenerations+1;
    elseif (convergecount >= parameters.genetictermination) && (parameters.geneticselection < 0.9)% && (parameters.geneticmutation < 0.9)
        parameters.geneticselection = min(0.9, parameters.geneticselection+0.1);
        parameters.geneticmutation = max(0.1, parameters.geneticmutation-0.1);
        convergecount = 0;
        disp(['Generation: ', num2str(gencount), '; Selection: ', num2str(parameters.geneticselection), '; Mutation: ', num2str(parameters.geneticmutation)]);
        [fit2, ensemblepop] = subfunc05(fit2, ensemblepop, parameters);
        disp(gencount); gencount = gencount + 1;
    elseif (convergecount < parameters.genetictermination) && (gencount < parameters.geneticgenerations)
        [fit2, ensemblepop] = subfunc05(fit2, ensemblepop, parameters);
        disp(gencount); gencount = gencount + 1;
    elseif (gencount >= parameters.geneticgenerations)
        results.ensembletime = toc(time1);
        results.ensembleconvergence = gencount;
        disp(gencount); gencount = gencount + 1;
    end
%     if (convergecount >= parameters.genetictermination)
%         results.ensembletime = toc(time1);
%         results.ensembleconvergence = gencount;
%         %gencount = parameters.geneticgenerations;
%     else
%         [fit2, ensemblepop] = subfunc05(fit2, ensemblepop, parameters);
%     end
    
    %save('matlab.mat');
end

% clear fit3
% fit3(1) = 0; fit3(2) = realmax;
% parfor (x = 1:nchrom, parameters.parallelcores)
%     [fit2(x, 1), newchrom, predarray1{x}, fitarray1{x}] = subfunc04(ensemblepop(x, :), preds, dataset(:, nfeats+1));
%     ensemblepop(x, :) = newchrom;
% end
% 
% results.classifierfit = fit1;
% results.classifierpop = population;
% results.classifierpreds = preds;
% results.ensemblefit = fit2;
% results.ensemblepop = ensemblepop;
% 
% for x = 1:nchrom
%     if (fit2(x, 1) > fit3(1))
%         fit3(1) = fit2(x, 1);
%         fit3(2) = count(ensemblepop(x, :)) < fit3(2);
%         predarray = predarray1{x};
%         fitarray = fitarray1{x};
%         results.bestensemblechrom = [x ensemblepop(x, :)];
%     elseif (fit2(x, 1) == fit3(1)) && (count(ensemblepop(x, :)) < fit3(2))
%         fit3(1) = fit2(x, 1);
%         fit3(2) = count(ensemblepop(x, :)) < fit3(2);
%         predarray = predarray1{x};
%         fitarray = fitarray1{x};
%         results.bestensemblechrom = ensemblepop(x, :);
%     end
% end
% output = ensemblevoting(predarray, dataset(:, nfeats+1), fitarray, 0);
% 
% results.bestensemblefit = fit3(1);
% results.bestensemblepredsall = predarray;
% results.bestensemblepreds = output;
% results.bestensemblefitarray = fitarray;

[results.bestensemblefit, a01] = max(fit2(:, 1));
a02 = find(ensemblepop(a01, 2:end) > 0);
results.bestensembleset = preds(a02, :);
results.bestensembleweights = fit1(a02, 1);
[results.bestensembleoutput, ~, ~, ~] = alexfastvote(results.bestensembleset', labels, results.bestensembleweights);

results.ensemblepopulation = ensemblepop;

%catch
    %disp('There is an error in mainhub2');
%end

save 10peo_6feat_100chro_1.mat results;

end % end mainfunction



function [fit2, pop2, preds2] = subfunc01(fit1, population, preds, parameters)
try
% genetic selection, reproduction, and mutation for FAM chromosomes
% set number of rejected chromosomes equal to size of a single subpopulation?
%  maximum, no minimum
subpopsize = round(parameters.chromosomes / parameters.HGA);
nsel = ceil(parameters.chromosomes * parameters.geneticselection);
nrej = parameters.chromosomes - nsel;
nrej = min(nrej, subpopsize);

[fit2(:, 1), order1] = sort(fit1(:, 1), 'descend');
subpops = fit1(order1, 2);
fit2(order1, 2) = subpops;
pop2 = population(order1, :);
pop3 = population(order1, :);
preds2 = preds(order1, :);

for x = 1:nrej
    
%for x = 1
    parent1 = []; parent2 = [];
    
    % choose a subpopulation and pick two random parents in it
    while (numel(parent1) == 0) || (numel(parent2) == 0)
        try
        a01 = randperm(max(fit2(:, 2)));
        a01 = a01(1);
        a03 = find(fit2(:, 2)==a01);
        a02 = randperm(numel(a03));
        
        parent1 = pop3(a03(a02(1)), :);
        parent2 = pop3(a03(a02(2)), :);
        catch
        end
    end
    
    % number of data samples to use for training
    %a01 = ceil((parent1{1} + parent2{1})/2);
    %a01 = max(10, a01);
    %a01 = min(parameters.ndata, a01);
    %offspring{1} = a01;
    
    % combine training sequence
    p01 = parent1{1};
    p02 = parent2{1};
    a01 = p01 == p02;
    a02 = 1:length(a01);
    a03 = zeros(1, length(a02));
    
    for y = 1:length(a02)
        if a01(y) == 1
            a03(y) = p01(y); % common gene in both parents
            a02(a03(y)) = 0; % remove gene from available pool
        elseif a01(y) == 0
            rand1 = rand * 2;
            if (rand1 <= 1) && (mean(find(a02 == p01(y))) > 0)
                a03(y) = p01(y);
                a02(a03(y)) = 0;
            elseif (rand1 > 1) && (mean(find(a02 == p02(y))) > 0)
                a03(y) = p02(y);
                a02(a03(y)) = 0;
            else
                a04 = find(a02 > 0);
                a03(y) = a04(1);
                a02(a03(y)) = 0;
            end
        end
    end
    offspring{1} = a03;
    
    % combine feature subset selection
    a01 = parent1{2};
    a02 = parent2{2};
    clear a03
    for y = 1:length(a01)
        if (a01(y) == a02(y))
            a03(y) = a01(y);
        else
            a03(y) = round(rand);
        end
    end
    while sum(a03) == 0
        rand1 = ceil(rand * length(a03));
        a03(rand1) = 1;
    end
    offspring{2} = a03;
    
    % ARTMAP parameters
    a01 = (parent1{3} + parent2{3}) / 2;
    offspring{3} = a01;
    
    % mutation
    nmut = round(length(offspring) * parameters.geneticmutation);
    for y = 1:nmut
        rand1 = ceil(rand * 3); % which segment to mutate
        %if rand1 == 1 
         %   a01 = offspring{1};
          %  rand2 = ((rand*2) - 1) * parameters.geneticmutation * a01;
            %a01 = a01 + round(rand2);
            %a01 = min(a01, parameters.ndata);
            %a01 = max(10, a01);
            %offspring{1} = a01;
            
         if rand1 == 1
            rand2 = ceil(rand * length(offspring{1}));
            rand3 = ceil(rand * length(offspring{1}));
            a01 = offspring{1};
            rand4 = a01(rand2);
            a01(rand2) = a01(rand3);
            a01(rand3) = rand4;
            offspring{1} = a01;
            
        elseif rand1 == 2
            rand2 = round(rand);
            rand3 = ceil(rand * length(offspring{2}));
            a01 = offspring{2};
            a01(rand3) = rand2;
            maxfeats = round(parameters.maxfeatures * parameters.nfeats);
            while sum(a01) == 0
                rand3 = ceil(rand * length(a01));
                a01(rand3) = 1;
            end
            while sum(a01) > maxfeats
                rand3 = ceil(rand * length(a01));
                a01(rand3) = 0;
            end
            offspring{2} = a01;
            
        elseif rand1 == 3
            a01 = offspring{3};
            rand2 = ceil(rand * length(a01));
            rand3 = ((rand*2) - 1) * parameters.geneticmutation;
            a01(rand2) = a01(rand2) + rand3;
            a01(rand2) = min(1, a01(rand2));
            a01(rand2) = max(0, a01(rand2));
            
            offspring{3} = a01;
            
        end
    end
    
    % replace chromosome in the lowest subpopulation with the lowest fitness
    % by alex, some bug
%    a01 = find(fit2(:, 2) == max(fit2(:, 2)));
%     a01 = find(fit2(:, 2) == max(fit2(:, 2)) & fit2(:, 1) >= 0);
%    a01 = find(fit2(:, 1) == min(fit2(:, 1)) & fit2(:, 1) >= 0);
    a01 = parameters.chromosomes - nrej + x;
    pop2(a01(end), :) = offspring;
    fit2(a01(end), 1) = -1;
    fit2(a01(end), 2) = max(fit2(:, 2));
    subpops(a01(end), 1) = max(fit2(:, 2));
    preds2(a01(end), :) = NaN;
    clear offspring parent1 parent2
    % by alex, some bug
    
    %Yuna
    %a01 = nsel + x;
    %pop2(a01, :) = offspring;
    %fit2(a01, 1) = 0;
    %preds2(a01, :) = NaN;
    %clear offspring parent1 parent2
    %Yuna
end

catch
    disp('Some error in subfunc01')
end

fit2 = [fit2(:, 1) subpops];
end % end subfunc01







function [a,b,c] = subfunc03(dataStruct, artparam)
inputNet = struct('M', {[]}, ... % Number of features
    'TestOnly', {0}, ... % Train & Test or Test only? unless specified otherwise
    'test_fast', {1}, ... % Use fast tester unless specified otherwise
    'compute_overlap', {1}, ... % Indirectly measure overlap by looking at how many nodes overlap with a test_point
    'C', {0}, ... % Number of committed coding nodes (paper says set to 0, but harder to Matlab)
    'plotSteps',{0}, ... % % Iterate through individual learning steps
    'NodeList', {[]}, ... % List of node associated with each input
    'maxNumCategories', {100}, ...
    'numClasses', {[]}, ... % The total number of classes
    'w', {[]}, ... % coding node weight vector
    'W', {[]}, ... % output class weight vector
    'rho_bar', {0}, ... % baseline vigilance; maximum code compression at 0, for faster algorithm
    'rho', {0}, ... % vigilance %Alex: no change in results; discard
    'base_vigilance', {0}, ... % Baseline vigilance; [0, 1], with 0 maximizing code compression %Alex: no change in results; discard
    'alpha', {.01}, ... % Signal Rule Parameter
    'alphaTrain',{[]},... % Store when reverting alpha value in TESTING
    'beta', {1.0}, ...  % Learning fraction; [0, 1], with 1 implementing fast learning
    'learningRate', {0}, ... %Alex: changes doesn't affect results; discard
    'gamma',{.000001},... % fraction additive in numerator and denominator of matching signal; .000001
    'epsilon', {-.001}, ... % Match Tracking (codes inconsistent cases); [-1, 1]; -.001
    'p', {1.0}, ... % CAM rule power; [0, inf], default 1.0
    'dataSubsets', {0}, ... % 4-fold cross-validation %Alex: changes doesn't affect results; discard
    'votes', {0}, ... % Number of voting systems: 5 %Alex: changes doesn't affect results; discard
    'win_sequence',{[]},...
    'act_sequence',{[]},...
    'search_cycles', {1},... % Total number of search cycles during training
    'lambda_Attention',{[0]},...
    'dataStore',{[]},... % Store data if using structs as inputs
    'numEpochs',{1},...%Number of times to present input
    'model_num',{[32]},... % Depletion Model Number
    'type_depletion',{4},... %Options are 0 1 2 3 4; %Alex: changes doesn't affect results; discard
    'deplete_fast',{[0]},...
    'learn_trail',{[]},...
    'learn_description',{''},...
    'short_learn_descript',{''},...
    'uncommitted_fail',{0},... %Alex: changes doesn't affect results; discard
    'NI_fail',{[]},...
    'NI_fail_addNode',{[]},...
    'node_created',{[]},...
    'start_conservative',{[0]},... %Alex: discard; slower training and convergence when set to 1
    'ovlp_net',{[0]},... %Alex: discard; slower training when set to 1
    'A_dep_dist',{[]},...
    'e_mod_dist',{[]},...
    'e_realLow',{0},...
    'e_zero',{0},...
    'diffE_count',{[]},...
    'perc_error',{[]},... % percentage error added to labels in training
    'var_error_features',{[]},... % variance of error added to features in training
    'testAccuracy',{[]},...
    'DEFAULT_RETEST', {1}); % Attention gain parameter

inputNet.numClasses = max(dataStruct.training_output); % assuming classes range from 1 to N
inputNet.alpha = artparam(1);
inputNet.beta = artparam(2);
inputNet.gamma = artparam(3);
inputNet.epsilon = artparam(4);
inputNet.rho_bar = artparam(5);

[a,b,c]=Default_ARTMAP(dataStruct, inputNet);
%     disp('Output class predictions are stored in variable ''a''.');
%     disp('Distributed output predictions are stored in ''b''.');
%     disp('The biased ARTMAP network details are stored in ''c''.');
end %end subfunc02










function [fitness, newchrom, predarray, fitarray] = subfunc04(chromosome, preds, labels)

% negative correlation

predarray = [];
fitarray = [];
fit3(1:2) = 0;
iter = 1;
newchrom(length(chromosome)) = 0;
newchrom(1) = chromosome(1);
while iter < (length(chromosome)-1)
    if (chromosome(iter+1) >= 0.1) && (isempty(predarray))
        predarray = preds(iter, :)';
        fitarray = chromosome(iter+1);
        
        fit3(1) = sum(predarray == labels) / length(labels);
        fit3(2) = 0;
        newchrom(iter+1) = chromosome(iter+1);
        chromosome(iter+1) = 0;
    elseif (chromosome(iter+1) >= 0.1) && (~isempty(predarray))
        output = ensemblevoting([predarray preds(iter, :)'], labels, [fitarray; chromosome(iter+1)], 0);
        
        acc = sum(output == labels) / length(labels);
        div = alex60([predarray preds(iter, :)']);
        
        ind1 = acc + (chromosome(1)*div);
        if (acc > fit3(1)) && (ind1 > fit3(2))
            fit3(1) = acc;
            fit3(2) = ind1;
            
            predarray = [predarray preds(iter, :)'];
            fitarray = [fitarray; chromosome(iter+1)];
            
            newchrom(iter+1) = chromosome(iter+1);
            chromosome(iter+1) = 0;
            iter = 1;
        end
    elseif (chromosome(iter+1) < 0.1)
        chromosome(iter+1) = 0;
    end
    
    iter = iter + 1;
end

% fitness based on overall accuracy
%  fitness based on ensemble size?
fitness = fit3(1);

end % end subfunc02



function [fit2, pop2] = subfunc05(fit1, population, parameters)

try
% genetic selection, reproduction, and mutation for ensemble chromosomes
nsel = ceil(parameters.chromosomes * parameters.geneticselection);
nrej = parameters.chromosomes - nsel;

[fit2(:, 1), fit2(:, 2)] = sort(fit1(:, 1), 'descend');
pop2 = population(fit2(:, 2), :);

for x = 1:nrej
    % select two random parents
    parent1 = population(ceil(rand * parameters.chromosomes), :);
    parent2 = population(ceil(rand * parameters.chromosomes), :);
    
    % average all the genes
    offspring = (parent1 + parent2) / 2;
    offspring = round(offspring);
    
    % mutation
    nmut = round(length(offspring) * parameters.geneticmutation);
    rmut = parameters.geneticmutation;
    for y = 1:nmut
        a01 = randperm(numel(offspring));
        offspring(a01) = 1 - offspring(a01);
    end
    
    if sum(offspring) == 0
        a01 = randperm(numel(offspring));
        a02 = ceil(rand*numel(offspring));
        offspring(a01(1:a02)) = 1;
    end
    
    % replace the bottom chromosomes
    pop2(nsel+x, :) = offspring;
    fit2(nsel+x, 1) = 0;
    clear offspring
end
fit2 = fit2(:, 1);

catch
    pause(0.001);
end

end % end subfunc01



function [fit2, pop2, preds2] = alex71(fit1, population, preds, nHGA)

% hierarchical subpopulations

fit3 = [];
% how many per HGA
nsize = size(fit1, 1);
npop = ceil(nsize / nHGA);
[fit2, order1] = sort(fit1(:, 1), 'descend');
pop1 = population(order1, :);
preds1 = preds(order1, :);
fit2(:, 2) = fit1(order1, 2);
% low HGA for high accuracy
% high HGA for low accuracy
for x = 1:nsize
    HGA = ceil(x / npop);
    if fit2(x, 2) == 0 % new unassigned chromosome
        fit2(x, 2) = HGA;
    elseif fit2(x, 2) < HGA 
        fit2(x, 2) = fit2(x, 2) + 1;
    elseif fit2(x, 2) > HGA
        fit2(x, 2) = fit2(x, 2) - 1;
    end
end

[fit3, order2] = sort(fit2(:, 2), 'ascend');
fit2 = fit2(order2, :);
pop2 = pop1(order2, :);
preds2 = preds1(order2, :);


end







