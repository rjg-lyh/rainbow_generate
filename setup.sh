# dvc init
# dvc remote add -d myremote s3://dvc
# dvc remote modify myremote endpointurl http://ceph01

set -x

# update git submodule

# download data
export AWS_ACCESS_KEY_ID=18M4BI7CGWKWSYELOXC3
export AWS_SECRET_ACCESS_KEY=zHpZOndT6y4Dkz5GkH4g1DAkyRuDKe7BUFSjbV3b

dvc pull


