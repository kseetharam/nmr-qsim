function out = path_fix()
spinachpath = '/n/holyscratch01/jaffe_lab/Everyone/kis/software/spinach_2_7_6009';
datapath = '/n/holyscratch01/jaffe_lab/Everyone/kis/data/spinach_data';

addpath(genpath(strcat(spinachpath,'/kernel')));
addpath(genpath(strcat(spinachpath,'/etc')));
addpath(genpath(strcat(spinachpath,'/experiments')));
addpath(genpath(strcat(spinachpath,'/interfaces')));

out = datapath;
end