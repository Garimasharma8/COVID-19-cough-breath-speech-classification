clc;
clear all;
close all;

%% covid cough 
addpath(genpath('/Users/garimasharma/Downloads/data-to-share-covid-19-sounds/KDD_paper_data/covidandroidwithcough/cough'));
for i = 1 : 204
  fname=strcat('CC',num2str(i));
audioname=strcat(fname,'.wav');
[x{i},fx{i}]=audioread(audioname);
x{i}= x{i}(:,1);  % monochannel
[sp{i}, fp{i},tp{i},pp{i}]= spectrogram(x{i}, 128,120,128,fx{i},'yaxis');
spectrogram(x{i}, 128,120,128,fx{i},'yaxis'); colorbar('off');
saveas(gcf,['CC',num2str(i),'.jpg']);
IC{i}=rgb2gray(imread(sprintf('CC%d.jpg',i)));
lbpp{i}=extractLBPFeatures(IC{i},'NumNeighbors',8,'Radius',3,'Upright',false,'CellSize',[256 256]);  % Rotation invariant LBP
lbp1{i}=extractLBPFeatures(IC{i},'radius',3,'Normalization','L2'); % to get 59 dimentional vector
glcm1{i}=graycomatrix(IC{i});
haralick1{i}= getHaralick(glcm1{i},[1:14]);
end
%% Covid non cough

addpath(genpath('/Users/garimasharma/Downloads/2021/data-to-share-covid-19-sounds/KDD_paper_data/covidandroidnocough/cough'));
for k=1:64
    fname=strcat('CNC',num2str(k));
audioname=strcat(fname,'.wav');
[x1{k},fx1{k}]=audioread(audioname);
x1{k}= x1{k}(:,1);  % monochannel
[sp1{k}, fp1{k},tp1{k},pp1{k}]= spectrogram(x1{k}, 128,120,128,fx1{k},'yaxis');
spectrogram(x1{k}, 128,120,128,fx1{k},'yaxis'); colorbar('off');
saveas(gcf,['CNC',num2str(k),'.jpg']);
IC1{k}=rgb2gray(imread(sprintf('CNC%d.jpg',k)));
lbpp1{k}=extractLBPFeatures(IC1{k},'NumNeighbors',8,'Radius',3,'Upright',false,'CellSize',[256 256]);  % Rotation invariant LBP
lbp11{k}=extractLBPFeatures(IC1{k},'radius',3,'Normalization','L2'); % to get 59 dimentional vector
glcm11{k}=graycomatrix(IC1{k});
haralick11{k}= getHaralick(glcm11{k},[1:14]);
end
    


%% Healthy cough

addpath(genpath('/Users/garimasharma/Downloads/data-to-share-covid-19-sounds/KDD_paper_data/healthyandroidwithcough/cough'));
for j=1:620
    fname=strcat('NCC',num2str(j));
audioname=strcat(fname,'.wav');
[y{j},fy{j}]=audioread(audioname);
y{j}=y{j}(:,1); % monochannel
[sh{j}, fh{j},th{j},ph{j}]= spectrogram(y{j}, 128,120,128,fy{j},'yaxis');
spectrogram(y{j}, 128,120,128,fy{j},'yaxis'); colorbar('off');
saveas(gcf,['NCC',num2str(j),'.jpg']);
INC{j}=rgb2gray(imread(sprintf('NCC%d.jpg',j)));
lbpnc{j}=extractLBPFeatures(INC{j}, 'NumNeighbors',8,'Radius',3,'Upright',false,'CellSize',[256 256]);  % full length LBP features
lbpncy{j}= extractLBPFeatures(INC{j},'radius',3,'Normalization','L2'); % to get 59 dimentional vector
glcm2{j}=graycomatrix(INC{j});
haralick2{j}=getHaralick(glcm2{j},[1:14]);
end


%% Healthy no cough
addpath(genpath('/Users/garimasharma/Downloads/2021/data-to-share-covid-19-sounds/KDD_paper_data/healthyandroidnosymp/cough'));
for m=1:138
    fname=strcat('HNS',num2str(m));
audioname=strcat(fname,'.wav');
[y2{m},fy2{m}]=audioread(audioname);
y2{m}=y2{m}(:,1); % monochannel
[sh2{m}, fh2{m},th2{m},ph2{m}]= spectrogram(y2{m}, 128,120,128,fy2{m},'yaxis');
spectrogram(y2{m}, 128,120,128,fy2{m},'yaxis'); colorbar('off');
saveas(gcf,['HNS',num2str(m),'.jpg']);
INC2{m}=rgb2gray(imread(sprintf('HNS%d.jpg',m)));
lbpnc2{m}=extractLBPFeatures(INC2{m}, 'NumNeighbors',8,'Radius',3,'Upright',false,'CellSize',[256 256]);  % full length LBP features
lbpncy2{m}= extractLBPFeatures(INC2{m},'radius',3,'Normalization','L2'); % to get 59 dimentional vector
glcmy2{m}=graycomatrix(INC2{m});
haralicky2{m}=getHaralick(glcmy2{m},[1:14]);
end

%% Asthma cough
addpath(genpath('/Users/garimasharma/Downloads/2021/data-to-share-covid-19-sounds/KDD_paper_data/asthmaandroidwithcough/cough'));
for n=1:104
     fname=strcat('AC',num2str(n));
audioname=strcat(fname,'.wav');
[y3{n},fy3{n}]=audioread(audioname);
y3{n}=y3{n}(:,1); % monochannel
[sh3{n}, fh3{n},th3{n},ph3{n}]= spectrogram(y3{n}, 128,120,128,fy3{n},'yaxis');
spectrogram(y3{n}, 128,120,128,fy3{n},'yaxis'); colorbar('off');
saveas(gcf,['AC',num2str(n),'.jpg']);
INC3{n}=rgb2gray(imread(sprintf('AC%d.jpg',n)));
lbpnc3{n}=extractLBPFeatures(INC3{n}, 'NumNeighbors',8,'Radius',3,'Upright',false,'CellSize',[256 256]);  % full length LBP features
lbpncy3{n}= extractLBPFeatures(INC3{n},'radius',3,'Normalization','L2'); % to get 59 dimentional vector
glcmy3{n}=graycomatrix(INC3{n});
haralicky3{n}=getHaralick(glcmy3{n},[1:14]);
end


%% try with lbp full feature

L1C= cell2mat(lbpp); L1C=L1C'; % COVID COUGH
L1C=reshape(L1C,[60,46]); L1C=L1C';

L1H=cell2mat(lbpnc); % healthy cough
L1H=L1H';
L1H=reshape(L1H,[60,64]); L1H=L1H';

L1A=cell2mat(lbpnc3); L1A=L1A'; %ASTHMA COUGH
L1A=reshape(L1A,[60,104]); L1A=L1A';


%L1NC=cell2mat(lbpnc); L1NC=L1NC'; L1NC=reshape(L1NC,[120,64]); L1NC=L1NC';
data1=[L1C;L1H;L1A];
group=zeros(214,1); group(47:110,1)=1; group(111:214,1)=2;

T1=table(data1,group);
% with PCA accuracy is 97.3%%


%% try with 59-dimentional LBP

L2C=cell2mat(lbp1); L2C=L2C'; L2C=reshape(L2C,[59,46]); L2C=L2C';
L2NC=cell2mat(lbpnc2); L2NC=L2NC'; L2NC=reshape(L2NC,[59,64]); L2NC=L2NC';
data2=[L2C;L2NC];
group2=zeros(110,1); group2(1:46,1)=1;

T2=table(data2,group2);
% with PCA accuracy is 94.5% 10fold CV

%% With haralick only, 14-haralick
% LBP for covid cough
%L1=[lbp1{1};lbp1{2};lbp1{3};lbp1{4};lbp1{5};lbp1{6};lbp1{7};lbp1{8};lbp1{9};lbp1{10};lbp1{11};lbp1{12};lbp1{13};lbp1{14};lbp1{15};lbp1{16};lbp1{17};lbp1{18};lbp1{19};lbp1{20};lbp1{21};...
    %lbp1{22};lbp1{23};lbp1{24};lbp1{25};lbp1{26};lbp1{27};lbp1{28};lbp1{29};lbp1{30};lbp1{31};lbp1{32};lbp1{33};lbp1{34};lbp1{35};...
    %lbp1{36};lbp1{37};lbp1{38};lbp1{39};lbp1{40};lbp1{41};lbp1{42};lbp1{43};lbp1{44};lbp1{45};lbp1{46}];
% LBP for normal cough
%L2= [lbpnc2{1};lbpnc2{2};lbpnc2{3};lbpnc2{4};lbpnc2{5};lbpnc2{6};lbpnc2{7};lbpnc2{8};lbpnc2{9};lbpnc2{10};lbpnc2{11};lbpnc2{12};lbpnc2{13};lbpnc2{14};lbpnc2{15};lbpnc2{16};lbpnc2{17};lbpnc2{18};lbpnc2{19};lbpnc2{20};lbpnc2{21};...
    %lbpnc2{22};lbpnc2{23};lbpnc2{24};lbpnc2{25};lbpnc2{26};lbpnc2{27};lbpnc2{28};lbpnc2{29};lbpnc2{30};lbpnc2{31};lbpnc2{32};lbpnc2{33};lbpnc2{34};lbpnc2{35};...
    %lbpnc2{36};lbpnc2{37};lbpnc2{38};lbpnc2{39};lbpnc2{40};lbpnc2{41};lbpnc2{42};lbpnc2{43};lbpnc2{44};lbpnc2{45};lbpnc2{46};lbpnc2{47};lbpnc2{48};...
    %lbpnc2{49};lbpnc2{50};lbpnc2{51};lbpnc2{52};lbpnc2{53};lbpnc2{54};lbpnc2{55};lbpnc2{56};lbpnc2{57};lbpnc2{58};lbpnc2{59};lbpnc2{60};...
    %lbpnc2{61};lbpnc2{62};lbpnc2{63};lbpnc2{64}];
% Haralaick for covid cough
H1= [haralick1{1};haralick1{2};haralick1{3};haralick1{4};haralick1{5};haralick1{6};haralick1{7};haralick1{8};haralick1{9};haralick1{10};haralick1{11};haralick1{12};haralick1{13};haralick1{14};haralick1{15};haralick1{16};haralick1{17};haralick1{18};haralick1{19};haralick1{20};haralick1{21};...
    haralick1{22};haralick1{23};haralick1{24};haralick1{25};haralick1{26};haralick1{27};haralick1{28};haralick1{29};haralick1{30};haralick1{31};haralick1{32};haralick1{33};haralick1{34};haralick1{35};...
    haralick1{36};haralick1{37};haralick1{38};haralick1{39};haralick1{40};haralick1{41};haralick1{42};haralick1{43};haralick1{44};haralick1{45};haralick1{46}];
H1=reshape(H1,[14,46])';

H2= [haralick2{1};haralick2{2};haralick2{3};haralick2{4};haralick2{5};haralick2{6};haralick2{7};haralick2{8};haralick2{9};haralick2{10};haralick2{11};haralick2{12};haralick2{13};haralick2{14};haralick2{15};haralick2{16};haralick2{17};haralick2{18};haralick2{19};haralick2{20};haralick2{21};...
    haralick2{22};haralick2{23};haralick2{24};haralick2{25};haralick2{26};haralick2{27};haralick2{28};haralick2{29};haralick2{30};haralick2{31};haralick2{32};haralick2{33};haralick2{34};haralick2{35};...
    haralick2{36};haralick2{37};haralick2{38};haralick2{39};haralick2{40};haralick2{41};haralick2{42};haralick2{43};haralick2{44};haralick2{45};haralick2{46};haralick2{47};haralick2{48};...
    haralick2{49};haralick2{50};haralick2{51};haralick2{52};haralick2{53};haralick2{54};haralick2{55};haralick2{56};haralick2{57};haralick2{58};haralick2{59};haralick2{60};...
    haralick2{61};haralick2{62};haralick2{63};haralick2{64}];
H2=reshape(H2,[14,64])';

Hdata=[H1;H2];
Hgroup=zeros(110,1); Hgroup(1:46,1)=1;
Htable=table(Hdata,Hgroup);

%% Conmibe full LBP and Haralick

codata=[L1C';H1'];
nocodata=[L1NC';H2'];
totaldata=[codata';nocodata'];
group2=zeros(110,1); group2(1:46,1)=1;

totaltable=table(totaldata,group2);

%% combine 59-d LBP and haralick

codata2=[L2C';H1'];
nocodata2=[L2NC';H2'];
totaldata2=[codata2';nocodata2'];
group2=zeros(110,1); group2(1:46,1)=1;
totaltable2=table(totaldata2,group2);


%% visulization of features

% LBP histogram error 1. Covid vs non-covid 2. Covid v/s covid
error1=(lbp1{1}-lbpnc2{1}).^2;
error2=(lbp1{1}-lbp1{2}).^2;
bar([error1;error2]','grouped')

% visualize by t-SNE and gscatter
% full LBP and haralick in 2D and 3D
%rng default
%Y=tsne(totaldata,'Algorithm','exact','Distance','mahalanobis');
%subplot(221)
%gscatter(Y(:,1),Y(:,2),group2)
%title('mahalanbois');

rng default
Y=tsne(totaldata,'Algorithm','exact','Distance','cosine');
subplot(221)
gscatter(Y(:,1),Y(:,2),group2)
title('cosine');

rng default
Y=tsne(totaldata,'Algorithm','exact','Distance','chebychev');
subplot(222)
gscatter(Y(:,1),Y(:,2),group2)
title('chebychev');

rng default
Y=tsne(totaldata,'Algorithm','exact','Distance','euclidean');
subplot(223)
gscatter(Y(:,1),Y(:,2),group2)
title('euclidean');

% another method for 2-D and 3-D
rng default
Y2=tsne(totaldata,'Algorithm','barneshut','NumPCAComponents',100);
figure; gscatter(Y2(:,1),Y(:,2),group2)

rng default
Y3 = tsne(totaldata2,'Algorithm','barneshut','NumPCAComponents',100,'NumDimensions',3);
figure
scatter3(Y3(:,1),Y3(:,2),Y3(:,3),15,group2,'filled');
view(-93,14)

%visualize actual time domain data points via t-sne












%% Classification by SVM

% full LBP+haralick 
% work on online version of matlab to get SVM plot
mdl=fitcsvm(totaldata,group2,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
sv=mdl.SupportVectors;
figure
gscatter(codata,nocodata,group2)
hold on
plot(sv(:,1),sv(:,2),'ko','MarkerSize',10)
hold off

[trainedClassifier, validationAccuracy] = trainClassifier(totaldata2)
%% COVID cough vs non-covid cough plots

subplot(211); plot(x{1}(6700:17800,1)); axis tight; subplot(212); plot(y{1}(7400:12800,1));axis tight;
subplot(121); spectrogram(x{1}(6700:17800,1),128,120,128,fx{1},'yaxis'); ...
    subplot(122); spectrogram(y{1}(7400:17800,1),128,120,128,fy{1},'yaxis');


%% tsne for LBP only

L1=cell2mat(lbpp); L1=reshape(L1,[120,204]); L1=L1';  %covid cough
L2=cell2mat(lbpnc); L2=reshape(L2,[120,620]); L2=L2';  % healthy cough
L3=cell2mat(lbpnc3);L3=reshape(L3,[120,104]); L3=L3'; % asthma cough
L4=cell2mat(lbpp1); L4=reshape(L4,[120,64]); L4=L4'; %covid no cough
L5=cell2mat(lbpnc2); L5=reshape(L5,[120,138]); L5=L5'; % healthy no cough
data=[L1;L2;L3;L4;L5];
group=zeros(416,1); group(47:110,1)=1; group(111:214,1)=2; group(215:278,1)=3;group(279:416,1)=4;
D1=[group, data];
Y = tsne(D1,'Algorithm','barneshut','NumPCAComponents',50,'perplexity',50);
figure
gscatter(Y(:,1),Y(:,2),group)

rng default % for fair comparison
Y3 = tsne(D1,'Algorithm','barneshut','NumPCAComponents',50,'NumDimensions',3);
figure
scatter3(Y3(:,1),Y3(:,2),Y3(:,3),15,group,'filled');
view(-93,14)   % accuracy is 79.1%

%% tsne for haralick only

H1=cell2mat(haralick1); H1=H1';
H2=cell2mat(haralick2); H2=H2';
H3=cell2mat(haralicky3);H3=H3';
H4=cell2mat(haralick11); H4=H4';
H5=cell2mat(haralicky2); H5=H5';
data1=[H1;H2;H3;H4;H5];
group=zeros(416,1); group(47:110,1)=1; group(111:214,1)=2; group(215:278,1)=3;group(279:416,1)=4;
D2=[group,data1];
Y = tsne(D2,'Algorithm','barneshut','NumPCAComponents',13,'perplexity',33);
figure
gscatter(Y(:,1),Y(:,2),group)

rng default % for fair comparison
Y3 = tsne(D2,'Algorithm','barneshut','NumPCAComponents',13,'NumDimensions',3);
figure
scatter3(Y3(:,1),Y3(:,2),Y3(:,3),15,group,'filled');
T3=table(data1,group); %accuracy is 65%

%% combine LBP and haralick
data2=[data, data1];
group=zeros(416,1); group(47:110,1)=1; group(111:214,1)=2; group(215:278,1)=3;group(279:416,1)=4;
D3=[group,data2];
Y = tsne(D3,'Algorithm','barneshut','NumPCAComponents',100,'perplexity',45);
figure
gscatter(Y(:,1),Y(:,2),group)

rng default % for fair comparison
Y3 = tsne(D3,'Algorithm','barneshut','NumPCAComponents',50,'NumDimensions',3);
figure
scatter3(Y3(:,1),Y3(:,2),Y3(:,3),15,group,'filled');
T4=table(data2,group); %accuracy is 76.7%


%% 5 class LBP only
L1=cell2mat(lbpp); L1= reshape(L1,[46,120]);
L2=cell2mat(lbpp1); L2= reshape(L2,[64,120]);
L3=cell2mat(lbpnc); L3= reshape(L3,[64,120]);
L4=cell2mat(lbpnc2); L4= reshape(L4,[138,120]);
L5=cell2mat(lbpnc3); L5= reshape(L5,[104,120]);

data=[L1;L2;L3;L4;L5];
group=zeros(416,1); group(47:110,1)=1; group(111:214,1)=2; group(215:278,1)=3;group(279:416,1)=4;

T=table(data,group);

%% LOOCV 
sse = 0; % Initialize the sum of squared error.
for i = 1:100
    [train,test] = crossvalind('LeaveMOut',length(data),1);
    yhat = polyval(polyfit(data(train),group(train),2),data(test));
    sse = sse + sum((yhat - group(test)).^2);
end

CVerr = sse / 100
