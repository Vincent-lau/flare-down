# flare-down
To what extent can we benefit from machine learning in terms of diagnosis and prognosis of diseases?

Introduction: 
The question being investigated in my Crest Project is regarding the application of artificial intelligence in the field of medical predication and treatment. In this project, efforts will be focused on building an algorithm that can facilitate families with little professional medical knowledge. 

A Well-posed definition by Tom Mitchell (1998) for machine learning is that: A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E. In other words, this is the process that the machine is training itself through large amount of dataset and adjust the parameters according to the measurement p to improve accuracy of its predications.

Since the middle of the last century, researchers have explored the potential applications of intelligent techniques in every field of medicine. (LB, 1955). The application of AI technology in the field of surgery was first successively investigated by Gunn in 1976, when he explored the possibility of diagnosing acute abdominal pain with computer analysis. (AA, 1976) So far, several systems have been developed to help analyse and process data such as Artificial neural networks, Fuzzy expert systems Evolutionary computation and Hybrid intelligent systems. Moreover, some projects carried out by universities and big companies are also on the way in this digital age. A team at Stanford University in California recently unveiled a machine-learning algorithm trained to scrutinise slides of cancerous lung tissue. The computer learned to pick out specific features about each slide, like the cells’ size, shape and texture. It could also distinguish between samples from people who had only lived for a short time after diagnosis – say, a few months – and ones from those who survived much longer. The study verified the algorithm’s results by testing it on historical data, so now the AI could in principle be used with patients. (Smart medicine is coming of age, but will doctors bite?, 2016) There are also examples where technology company is working with national authority to achieve better medical analysis. Google DeepMind, based in London, is using masses of anonymised data from the UK’s National Health Service to train an AI that will help ophthalmologists. The aim here is to spot looming eye disease earlier than a human can. (Smart medicine is coming of age, but will doctors bite?, 2016)

Apart from efforts on serious diseases, applications that are designed for people to diagnose their health conditions have also been made possible by scientists and doctors in recent years. A range of choices including Ada, Dr. AI and YourMD all aim to make credible medical advice more accessible and lead you through a clinical-grade triage process by using the principle of AI. (An app a day keeps the doctor away, 2017) These applications are trained by experts in the fields of both medicine and AI to make predications as accurate as possible and increasingly precise by learning themselves with multiple real-life situations. There are various benefits of using these apps: (1) users are able to have a brief concept of the seriousness and receive suggestions for further steps to take; (2) reducing the workload on doctors by offering patients the choice of consulting AI; (3) machines’ performance will not decrease over time but humans do need rest for better efficiency.

Despite all the progress that has been made so far in medicine using machine learning, there remains some problems. For example, although there are lots of software that can help people diagnose their existing diseases, there seems to be few tools that can tell people in advance what symptoms they might have and what corresponding precautions need to be taken based upon users’ daily habits and what conditions they are suffering from. Output given by the program might only be suggestions at the beginning but with increasingly more interactions with users and the program, the advice could be more and more accurate. Therefore, improvements need to be made on how to give people light warning before the actual symptom occurs. 

Method outline:
To begin with, I looked for dataset from different sources, aiming to find a dataset that includes people’s everyday activities and any diseases that they are suffering. After some thorough checking, I decided to use this dataset obtained from Kaggle(https://www.kaggle.com/flaredown/flaredown-autoimmune-symptom-tracker/data), and the original provider of this dataset is Flaredown (http://flaredown.com/). For the full dataset, please refer to the appendix or the websites stated above. Here is an extract of the dataset:
user_id,age,sex,country,checkin_date,trackable_id,trackable_type,trackable_name,trackable_value
QEVuQwEABlEzkh7fsBBjEe26RyIVcg==,,,,2015-11-26,1069,Condition,Ulcerative colitis,0
QEVuQwEAWRNGnuTRqXG2996KSkTIEw==,30,male,US,2015-11-26,1069,Condition,Ulcerative colitis,0
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3168,Condition,pain in left upper arm felt like i was getting a shot,4
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3169,Condition,hip pain when gettin up,3
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3170,Condition,pain in hand joints,4
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3171,Condition,numbness in right hand,2
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,1356,Condition,Headache,2
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3172,Condition,pain in left ankle,1
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3173,Condition,pain in left leg,1
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3174,Condition,pain in joints on feet,2
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3175,Condition,neck and upper back pain,2
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3176,Condition,neck pain at base of scull,2
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3177,Condition,pain inside arm and around elbow,2
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3178,Condition,diziness,3
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,3179,Condition,pain in face and jaw,1
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,123,Symptom,Joint stiffness,3
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,9363,Weather,icon,rain
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,9363,Weather,temperature_min,54.0
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,9363,Weather,temperature_max,69.0
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,9363,Weather,precip_intensity,0.0031
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,9363,Weather,pressure,1012.0
QEVuQwEA+WkNxtp/qkHvN2YmTBBDqg==,0,female,CA,2017-04-28,9363,Weather,humidity,66.0
QEVuQwEAHgM/igE3w0tBL14Jq1nEfw==,,,,2015-06-22,269,Condition,Crohn's disease,3
QEVuQwEAO+R1md5HUn8+w1Qpbg7ogw==,,,,2015-05-26,421,Condition,Gastroparesis,2
QEVuQwEAO+R1md5HUn8+w1Qpbg7ogw==,,,,2015-05-26,152,Symptom,Nausea,1
QEVuQwEAO+R1md5HUn8+w1Qpbg7ogw==,,,,2015-05-26,9890,Treatment,Zofran,8.0 mg
QEVuQwEAO+R1md5HUn8+w1Qpbg7ogw==,,,,2015-05-26,1,Tag,tired,
QEVuQwEAO+R1md5HUn8+w1Qpbg7ogw==,,,,2015-05-26,2,Tag,stressed,
QEVuQwEAO+R1md5HUn8+w1Qpbg7ogw==,,,,2015-05-26,3,Tag,feels,
QEVuQwEAO+R1md5HUn8+w1Qpbg7ogw==,,,,2015-06-14,421,Condition,Gastroparesis,0

A brief explanation from the website about the data is the following:
Instead of coupling symptoms to a particular illness, Flaredown asks users to create their unique set of conditions, symptoms and treatments (“trackables”). They can then “check-in” each day and record the severity of symptoms and conditions, the doses of treatments, and “tag” the day with any unexpected environmental factors.
User: includes an ID, age, sex, and country.
Condition: an illness or diagnosis, for example Rheumatoid Arthritis, rated on a scale of 0 (not active) to 4 (extremely active).
Symptom: self-explanatory, also rated on a 0–4 scale.
Treatment: anything a patient uses to improve their symptoms, along with an optional dose, which is a string that describes how much they took during the day. For instance “3 x 5mg”.
Tag: a string representing an environmental factor that does not occur every day, for example “ate dairy” or “rainy day”.
Food: food items were seeded from the publicly-available USDA food database. Users have also added many food items manually.
Weather: weather is pulled automatically for the user's postal code from the Dark Sky API. Weather parameters include a description, precipitation intensity, humidity, pressure, and min/max temperatures for the day.
If users do not see a symptom, treatment, tag, or food in our database (for instance “Abdominal Pain” as a symptom) they may add it by simply naming it. This means that the data requires some cleaning, but it is patient-centered and indicates their primary concerns.

There are some problems regarding this dataset, which needs to be handles before machine learning algorithms can be applied. For example, some bits of information are missing from certain users, and most of the information is in the form of a string, which is not suitable for machine learning as numbers are usually preferred. Therefore, it is essential to pre-process the data so that the efficiency of the algorithm can be ensured.

After proper processing is carried on the raw dataset, machine learning algorithms will be used on the reformatted dataset. There are lots of options available in terms of what machine learning algorithm to use, and it is hard to tell in advance which algorithm is the best one for this dataset. I decided to use neural network due to two facts listed below: 
•	firstly, with increasing computing power, neural network is now able to learn more complex patterns from the data in smaller amount of time; 
•	secondly, as the digital age arrives, a lot more data is produced every day than in the past, which is exactly what neural network needs. 
The actual used in this project will both be implemented by myself and downloaded from the Python library Scikit-learn (Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.)

After algorithms are applied to the data, results can be analysed and performance of the algorithm may be improved using several methods, such as tuning network, namely, improving neural network performance by changing parameters of the algorithm, and more data can be included to obtain better performance as well. More details of how to tune the network can be found in the results and analysis. For the source code, please refer to the appendix.

There are few hazards or safety issues as all data collected from users by Flaredown are anonymous. For possibly wrong predications of diseases, we do not need to take this into consideration until the program is published and used by customers.

 Results and Analysis:
As machine learning cannot have significant improvement without trial and error, I have altogether six versions of the program to demonstrate the process of analysing results and making adjustments accordingly. Because there are two main steps involved in my program, which are processing data and applying the algorithm. I will be mainly talking about these two steps and the differences from one version to another.

Version1:
I decided to keep the first version of my algorithm quick and dirty. Considering there are too many different types of conditions, tags and symptoms, I analysed the frequency of those “trackable types” and selected those types with the highest frequencies only. For example, if a condition only appears once in the whole dataset, then it will be discarded and not used by the algorithm. Data that are incomplete will be removed as well.

Because the objective of the program is to make predications on symptoms by taking conditions and tags as input, the raw data is consequently grouped when a symptom appears. Concretely, a symptom is used as the y-value and all tags and conditions preceding it will be considered as x-value. For the actual result, please to Version1/rearrangement.txt in the appendix. 

Conventionally, neural network will have same number of inputs for every sample. Due to this restriction, some data are further discarded. Only the two most frequent conditions and tags are left in the dataset and if there are fewer than two conditions or tags, then the whole sample will not be used. Here is a snapshot that shows the final result of the processed dataset:

 
After all of these reformatting, the data are ready to be converted into numbers for machine to learn from them. The way I converted them to numbers is that every group will become one sample, and other variables, such as age, gender or countries will all be mapped to a corresponding number which will either be a cardinal number that stands for how many of something there are such as how severe a condition is or an ordinal number which identifies the presence of a certain status, such as being tired. Here is a peek of the result:
  

Finally, neural network is applied to the data. This algorithm is adapted from https://hub.courseranotebooks.org/user/jyvluiohpnatetiyufkfkp/notebooks/Week%203/Planar%20data%20classification%20with%20one%20hidden%20layer/Planar%20data%20classification%20with%20one%20hidden%20layer%20v5.ipynb and implemented by myself. Because it is the first neural network, it does not have many hyperparameters and there is one hidden layer with the size of 4. The output function is chosen to be sigmoid function. After 10000 iterations, the algorithm’s performance on the train and set are:
 
Where cost is computed by the Cross-entropy loss function, which is a way of measuring the loss of a classification algorithm. And the performance is calculated based on how many correct predications there are in all cases.

It can be easily seen that the accuracy is not satisfactory, apart from this, there are lots of drawbacks in version1, for example, lots of information is ignored and many parameters in the neural network is not properly adjusted.

Version2:
Before tackling the problem that not enough information is included in the dataset, I also found out that the original dataset is not ordered chronologically, but clearly that an activity will normally contributes to the symptom afterwards, therefore it is important that data is arranged by time. 

I also noted that there are lots of symptoms that are consecutive, namely, there are multiple y-value that I needs to output. In order to keep the algorithm simple at the beginning, I eliminate all symptoms that have lower severity and keep the one with highest severity as it might be easier to for the algorithm to predict.

Next, I tried to include all “trackable types” in the raw dataset. There are many ways of doing this. The method I used, taking condition as an example, is to produce a list that consists of all conditions in the dataset, and then assign every condition a unique ordinal number. In this way, a condition vector can be created. If we have 200 conditions altogether, then a vector of size 200 is defined. If a particular condition is present in a sample, then the appropriate index of the vector will be assigned to the severity of that condition. For example, if the condition “hip pain when getting up” corresponds to the 1st condition and the severity of “hip pain when getting up” is 3, then the first item of the vector will be equal to 3. If a condition is not present, then it will remain zero. 
The operation is carried out on tags and symptoms, though the vector of them will only consist of zeros and ones to identify whether something exists or not.
 
In this way, every single variable will be included in the processed data, leading to a dataset that is too large to be trained in a reasonable amount of time. I have to reduce then number of samples in order to speed up the machine learning algorithm. 

 
Final results obtained from this version is even worse than the first version (shown below). Possible causes might include: there are too many factors that are affecting the algorithm, some of these might be negatively affecting the algorithm since some of these factors only appear several times or even once that they should actually have been ignored. Additionally, neither the dataset nor the algorithm is complex enough to learn from such a large dataset.
 Version3:
Having seen the effects of a too large or too small dataset, I realised that it is crucial to ensure that the size of the training set is chosen carefully. Considering that most of these data are typed by users of the app Flaredown themselves, it would be worthwhile to use natural language processing as part of the data processing. The pre-trained word vectors used in NLP is downloaded from http://bio.nlplab.org/ which provides various language resources created from the entire available biomedical scientific literature, a text corpus of over five billion words. Please refer to PMC-w2v in the appendix for full details of the word vector. 

After word vectors are obtained, every “trackable-name”, represented as a string, will be averaged into a single word vector. For instance, every word in the condition “pain in face and jaw” will have its own word vector, and the final word vector used in the algorithm will be the average value of these five vectors. This process turns all the natural language typed by the user into numbers, which are better understood by computers. However, this still doesn’t solve the problem that there are too many inputs. 

Clustering algorithm can be used here to tackle this issue. Specifically, another machine learning algorithm called k-means clustering is used here in order to group different “trackable-names”. The point of doing this is that, for example, the condition “insomnia” and “sleep disorder” could possibly be grouped together and treated the same by the algorithm. However, a new problem caused by using k-means is that number of clusters has to be determined by the programmer. This is not a straightforward task to accomplish since it requires lots of medical knowledge to see whether the clustering is reasonable or not and it does add more complexity to the program which is beyond the aim of this project. But by looking at the graph of error against number of clusters, I chose the “elbow point” provisionally where the tangent starts to slow down as this is how the choice is made conventionally.
 

 
 


When it comes to measuring the performance of this new method, I adopted a new measurement to estimate the accuracy. As the y-value is a vector that comprises of only one 1 and all others are 0s, a function that compares every element in the output and records every correct predication made by the algorithm and the percentage of correct predications can be calculated. The result is unexpectedly high, which are:
 
Despite this high performance on both the training and test set, it is still worth considering the number of 0s and 1s in the actual y-value. Since the actual y-value is dominated by zeros, we can easily obtain a very high performance even if the output is always zero! Due to this reason, another method to properly measure the performance needs to be used to deal with this biased data.

Version4:
The most common to handle biased data is to introduce precision and recall to give an indication of how well the algorithm is doing. The definitions of precision and recall are: (picture from https://en.wikipedia.org/wiki/Precision_and_recall )
 
Apart from using new estimation of the performance, there are also two other improvements in this version. In version2, I only kept the symptom with highest severity among consecutive symptoms. In fact, neural network algorithm is able to do multi-task learning, which can identify more than one symptoms. This can include more information from the raw dataset rather than simply choosing the one with highest value, which can potentially make the model more accurate. The final improvement is that some samples in the dataset are not good, for example:
  
These two samples only have output but no input, therefore it is reasonable to discard them to improve the performance of the algorithm.

After all of the improvements are carried out, I obtained the following result:

 
With two numbers in the bracket standing for precision and recall respectively.

Performance on test set decreased a lot compared with the last version, but it is a better representation which gives better estimation. However, further improvements can still be made.

Version5:
Before tuning neural network, a useful technique is applied in order to allow the algorithm to “see” more data before being tested on the new set. I shuffled the dataset before splitting it into training and test set so that we need to worry that the algorithm will under-perform due to the devoid of comprehensive data we feed into it. And it does prove to be useful according to the new result:

 

Once shuffling is done, I started to tune the hyperparameters of my neural network. The main hyperparameter that I need to modify are: regularization term-alpha, hidden layer size and learning rate of the algorithm. Here is the process of tuning these three parameters:
Hidden layer size:
By looking at the performance figure in version4, it can be clearly seen that the algorithm is overfitting the dataset, therefore smaller number of layers needs to be used. I will stick to only one hidden layer since number of samples in my dataset is only about 6000, which is not huge. After making this decision, I need to adjust number of hidden units to achieve best performance. The graph is plotted:
 

It is not difficult to see that when number of hidden units is equal to 300, the algorithm has the best performance on the dev test, therefore the number of hidden units for the first and the only layer is chosen to be 300.

Regularization term alpha:
By applying a similar approach, the following graph is obtained:
 
Again, the scatter show that, on the dev set (the orange points), the algorithm is best performed when alpha=1.00, so the 1 is used as hyperparameter alpha.
Learning rate:
The last parameter to be tuned is the learning rate and the graph is as shown:
 
The performance of the program on the dev set declines as the learning rate increases. However, first few points are very close together. Therefore, considering the running time of the program, a small value of the learning rate is chosen, which is 0.001.
After the tuning process is done, new values for the performance on the train and dev test is obtained again:
 
The improvement on the overall performance is around 4%, which is not huge. However, a major improvement on the precision on the dev set can be seen: from 63% to 73%.

Extra Version:
This is an extra version for deciding number of clusters for all conditions, tags and symptoms. In k-means algorithm, “elbow point” is conventionally chosen to be the number of clusters to be used. However, in this particular case, it might be worthwhile to consider whether the clustering result makes medical sense, i.e. will a human doctor do the clustering in a similar way if given a certain number of conditions? Unfortunately, I do not have access to such a medical expert to help me the sensibility of clustering algorithm, therefore, I carried out a test on the performance of the algorithm with different number of clusters.
The range of values for clusters I tried is [40,100]. The impact of the number of clusters on the performance is not significant, as shown by the result of plugging different values into the algorithm (performance ranging from 54% to 66%). Details of this tuning can be found in the appendix, here is an extract.
 

Result

This test is not particularly relevant to the original aim of my project, and it considered to be an extension to other versions in order to make the project closer to the real world. It is also an important improvement that could be made which will be discussed in more detail in the Discussion section.


A brief summary of all six versions of my program: the program starts from simple by ignoring some features with less importance and builds up the complexity by processing natural language as well as clustering similar features and discarding data that are not helpful for the algorithm to learn the pattern. The final achievement is tested on a test set by reshuffling the whole dataset in a different order to prevent overfitting, and the performance achieved is:
 









Discussion:
As summarised above, the final performance on a test set achieved by the algorithm is approximately 56%, which is not an ideal value. However, when estimating the performance of a machine learning algorithm, it is crucial to compare it with the so-called human level performance, namely, how well a human expert, or a panel of experts can do on this task. Again, it would require a lot more investments to accomplish the comparison between the machine and experts which is not accessible to my project. Despite this limitation, 56% is a respectable result to give people some reference.

Size of data:
Machine learning algorithm usually requires very large amount of data to achieve a good performance. Generally. having more data will always result in a better accuracy. The dataset I used is not particularly large considering the huge amount of data collected by large IT companies from people’s daily online activities. The quality of the data is not guaranteed either due to the need to modify them in order to make it more suitable for the algorithm to run on. Moreover, some samples in the dataset with very low quality is also discarded to prevent the algorithm to learn from such “outliers”. 

What other improvements that could have been made:
There are some other methods in neural network that could have been used, such as the using recurrent neural network to deal with the natural language processing which enable the program to take the order of words into consideration. Different models of machine learning algorithms could also be adopted such as the SVM algorithm and so on, which cannot learn such complex patterns as neural network but could perform well on this particular problem. Last but not least, some features have always been ignored in the processing such as “food” or “weather” due to the small number of presence. If the data collected has a larger proportion of these features, they could be included to give algorithms more information, which might give a positive feedback as well.

Implications:
Although the result of my algorithm is not very satisfying, it is undoubtable that machine learning will thrive in medicine. Future research efforts could be put into collecting professional medical data by large organisations as well as seeking assistance from medical experts in what parameters to include in the algorithm to make its predications more reliable and accurate. With all that being said, though it is not applicable to produce an algorithm that is very medically beneficial individually, it is still very likely for large companies to accomplish this task due to the fact that improvements can be seen over all versions of a very simple program with some data that are not professional.




  





















