# taken from: https://towardsdatascience.com/download-city-scapes-dataset-with-script-3061f87b20d7

DOWNLOAD_PATH='code/data/cityscapes'
COOKIES_FILE='cookies.txt'

#packageIDs map like this in the website:
#1 -> gtFine_trainvaltest.zip (241MB)
#2 -> gtCoarse.zip (1.3GB)
#3 -> leftImg8bit_trainvaltest.zip (11GB)
#4 -> leftImg8bit_trainextra.zip (44GB)
#8 -> camera_trainvaltest.zip (2MB)
#9 -> camera_trainextra.zip (8MB)
#10 -> vehicle_trainvaltest.zip (2MB)
#11 -> vehicle_trainextra.zip (7MB)
#12 -> leftImg8bit_demoVideo.zip (6.6GB)
#28 -> gtBbox_cityPersons_trainval.zip (2.2MB)

printf "Cityscapes package id:"
read PACKAGE_ID

printf "Cityscapes user:"
read CITY_USER

stty -echo
printf "Cityscapes password:"
read CITY_PASS
stty echo
printf "\n"

mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH

wget --keep-session-cookies --save-cookies=$COOKIES_FILE --post-data "username=$CITY_USER&password=$CITY_PASS&submit=Login" https://www.cityscapes-dataset.com/login/
wget --load-cookies $COOKIES_FILE --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=$PACKAGE_ID

rm index.html
rm $COOKIES_FILE
