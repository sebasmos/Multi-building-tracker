
#  The SpaceNet 7 Baseline Algorithm 
-----------

1. Download SpaceNet 7 Data
    
        cd /local_data/sn7/aws_download/
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz .
        aws s3 cp s3://spacenet-dataset/spacenet/SN7_buildings/tarballs/SN7_buildings_test_public.tar.gz .

    Extract data from the tarballs:
    
         cd /local_data/sn7/aws_download/
         tar -xvf SN7_buildings_train.tar.gz
         tar -xvf SN7_buildings_test_public.tar.gz


2. Download SpaceNet 7 baseline code to the desired location (e.g. `/path_to_baseline/`):
    
        cd /path_to_baseline/
        git clone https://github...
 
3. In Ubuntu 20.10: 
Install anaconda https://docs.anaconda.com/anaconda/install/linux/ 


Install solaris: https://solaris.readthedocs.io/en/latest/installation.html 


Install Nvidia-docker https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html  


Install docker: https://docs.docker.com/get-docker/  

After this, create a new Solaris environment in ANaconda:

```git clone https://github.com/cosmiq/solaris.git```

```cd solaris```

```conda env create -f environment-gpu.ymlv```

```conda activate solaris```

```pip install . ```

From console, always activate this option manually through``` conda activate solaris ``` or manually tipe ```anaconda-navigator``` from terminal and change ```root``` to ```solaris```, install Jupyter in this environment +  requirements.txt using ```pip install -r requirements.txt```. 
