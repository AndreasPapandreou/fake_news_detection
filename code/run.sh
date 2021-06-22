echo -e "---------------------------------"
echo -e "Process started!"
echo -e "---------------------------------\n"

# 0. Download data
echo -e "\n---------------------------------"
echo -e "downloading data..."
echo -e "---------------------------------\n"
sleep 2

mkdir ../data
mkdir ../general
wget -O ../data/data.zip wget https://www.dropbox.com/s/euudgdebmy7azq5/data.zip?dl=0
wget -O ../general/general.zip wget https://www.dropbox.com/s/6nll7kqmyschcuq/general.zip?dl=0
unzip ../data/data.zip -d ../data
unzip ../general/general.zip -d ../general

# 1. install some dependencies:

echo -e "\n---------------------------------"
echo -e "installing dependencies..."
echo -e "---------------------------------\n"
sleep 2
sudo apt-get install python-pip
sleep 2
pip install tensorflow
pip install Theano
pip install numpy scipy
pip install scikit-learn
pip install pillow
pip install h5py
pip install keras
pip install numpy
pip install psycopg2
pip install nltk
sleep 5
sudo apt-get update

echo -e "\n---------------------------------"
echo -e "setting env variables..."
echo -e "---------------------------------\n"
sleep 2

# set default configuration values
typeset -A config
config=(
    [POSTGRES_DB]="postgres"
    [AUTHORS_DB]="authors"
    [POSTGRES_USER]="postgres"
    [POSTGRES_PASSWORD]="postgres"
    [POSTGRES_HOST]="127.0.0.1"
    [POSTGRES_PORT]="5432"
    [GLOVE_DIR]="../../general/glove.6B"
    [GLOVE_OPTION]="glove.6B.50d.txt"
    [EMBEDDING_DIM]="50"
    [DROPOUTPER]="0.1"
    [HAN1_EPOCHS]="1000"
    [FHAN3_PRETRAIN_EPOCHS]="5000"
    [BATCH_SIZE]="1"
    [LEARNING_RATE]="0.00001"
    [DECAY]="1e-6"
    [MOMENTUM]="0.9"

    # the numbres 18 and 27 have been computed for the current dataset
    [AVER_WORDS]="18"
    [AVER_SENTENCES]="27"
)

# get the configuration values from the conf file
POSTGRES_DB=${config[POSTGRES_DB]}
AUTHORS_DB=${config[AUTHORS_DB]}
POSTGRES_USER=${config[POSTGRES_USER]}
POSTGRES_PASSWORD=${config[POSTGRES_PASSWORD]}
POSTGRES_HOST=${config[POSTGRES_HOST]}
POSTGRES_PORT=${config[POSTGRES_PORT]}
GLOVE_DIR=${config[GLOVE_DIR]}
GLOVE_OPTION=${config[GLOVE_OPTION]}
EMBEDDING_DIM=${config[EMBEDDING_DIM]}
DROPOUTPER=${config[DROPOUTPER]}
HAN1_EPOCHS=${config[HAN1_EPOCHS]}
FHAN3_PRETRAIN_EPOCHS=${config[FHAN3_PRETRAIN_EPOCHS]}
BATCH_SIZE=${config[BATCH_SIZE]}
LEARNING_RATE=${config[LEARNING_RATE]}
DECAY=${config[DECAY]}
MOMENTUM=${config[MOMENTUM]}
AVER_WORDS=${config[AVER_WORDS]}
AVER_SENTENCES=${config[AVER_SENTENCES]}

sleep 2

echo '' | sudo tee --append /etc/environment
echo "export POSTGRES_DB=$POSTGRES_DB" | sudo tee --append /etc/environment
echo "export AUTHORS_DB=$AUTHORS_DB" | sudo tee --append /etc/environment
echo "export POSTGRES_USER=$POSTGRES_USER" | sudo tee --append /etc/environment
echo "export POSTGRES_PASSWORD=$POSTGRES_PASSWORD" | sudo tee --append /etc/environment
echo "export POSTGRES_HOST=$POSTGRES_HOST" | sudo tee --append /etc/environment
echo "export POSTGRES_PORT=$POSTGRES_PORT" | sudo tee --append /etc/environment
echo "export GLOVE_DIR=$GLOVE_DIR" | sudo tee --append /etc/environment
echo "export GLOVE_OPTION=$GLOVE_OPTION" | sudo tee --append /etc/environment
echo "export EMBEDDING_DIM=$EMBEDDING_DIM" | sudo tee --append /etc/environment
echo "export DROPOUTPER=$DROPOUTPER" | sudo tee --append /etc/environment
echo "export HAN1_EPOCHS=$HAN1_EPOCHS" | sudo tee --append /etc/environment
echo "export FHAN3_PRETRAIN_EPOCHS=$FHAN3_PRETRAIN_EPOCHS" | sudo tee --append /etc/environment
echo "export BATCH_SIZE=$BATCH_SIZE" | sudo tee --append /etc/environment
echo "export LEARNING_RATE=$LEARNING_RATE" | sudo tee --append /etc/environment
echo "export DECAY=$DECAY" | sudo tee --append /etc/environment
echo "export MOMENTUM=$MOMENTUM" | sudo tee --append /etc/environment
echo "export AVER_WORDS=$AVER_WORDS" | sudo tee --append /etc/environment
echo "export AVER_SENTENCES=$AVER_SENTENCES" | sudo tee --append /etc/environment

sleep 2

source /etc/environment

# 
# 2. Install postgres
# 
echo -e "\n---------------------------------"
echo -e "installing postgres..."
echo -e "---------------------------------\n"
sleep 2
sudo apt install -y postgresql postgresql-contrib

# 
# 3. Add all data in postgres
# 
cd ../database
sudo service postgresql restart
bash ./load_data.sh
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
sudo service postgresql restart

# #
# 4. Run the code
# #
echo -e "\n---------------------------------"
echo -e "running the code..."
echo -e "---------------------------------\n"
sleep 2

# Build our model
echo -e "\n---------------------------------"
echo -e "building our model..."
echo -e "---------------------------------\n"
sleep 2
cd ../code/model_creation
python callerHan1.py &&
python prepareModel.py

# Evaluate your model
echo -e "\n---------------------------------"
echo -e "evaluating our model..."
echo -e "---------------------------------\n"
sleep 2
python evaluateModel.py

# Make some predictions
echo -e "\n------------------------------------------"
echo -e "making some predictions using our model..."
echo -e "------------------------------------------\n"
sleep 2
python predictModel.py

# Score authors using model A
echo -e "\n---------------------------------"
echo -e "score authors using model A..."
echo -e "---------------------------------\n"
sleep 2
cd ../score_authors
python scoreAuthorsModelA.py

# Score authors using model A and B
echo -e "\n-------------------------------------"
echo -e "score authors using model A and B..."
echo -e "-------------------------------------\n"
sleep 2
python scoreAuthorsModelA_and_B.py

# Αfter we have our model and all scores per author, we now can
# predict the authenticity of some new articles using once our model A 
# and later both of models A and B.
echo -e "\n--------------------------------------------"
echo -e "score new articles using only model A..."
echo -e "--------------------------------------------\n"
sleep 2
cd ../final_test
python scoreArticlesModelA.py

echo -e "\n---------------------------------------------"
echo -e "score new articles using models A and B..."
echo -e "---------------------------------------------\n"
sleep 2
python scoreArticlesModelA_and_B.py

echo -e "\n---------------------------------"
echo -e "Process completed!"
echo -e "---------------------------------\n"
