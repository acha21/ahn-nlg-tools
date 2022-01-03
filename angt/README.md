
## Install prerequisites for evaluation Reddit dataset
 * You need to install the following perl modules (e.g. by cpan install) XML:Twig (if you use docker ubuntu, you may install libxml-parser-perl), Sort:Naturally and String:Util.
 * Also you requires Java.
```bash
$ sudo apt-get install -y unzip unrar p7zip-full libxml-parser-perl
$ cpan install Sort:Naturally String:Util XML:Twig
```

### Download 3rd party codes for evaluation (as suggested in [CMR's code repository](https://github.com/qkaren/converse_reading_cmr/tree/master/evaluation))

* Please **downloads** the following 3rd-party packages and save in a new folder `3rdparty`:
	* [**mteval-v14c.pl**](https://drive.google.com/u/0/uc?id=1xsfykcaxkbhoZI2xbUpxGJZ7SmGIx_xc&export=download) to compute [NIST](http://www.mt-archive.info/HLT-2002-Doddington.pdf). 
	* [**meteor-1.5**](http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz) to compute [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/index.html). It requires [Java](https://www.java.com/en/download/help/download_options.xml).
* You can download the scripts as follows:
```bash
wget --no-check-certificate 'https://drive.google.com/u/0/uc?id=1xsfykcaxkbhoZI2xbUpxGJZ7SmGIx_xc&export=download' -O mteval-v14c.pl
wget --no-check-certificate 'http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz' -O meteor-1.5.tar.gz
```
* Move mteval-v14.pl and meteor-1.5.tar.gz to data/3rdparty and extract meteor-1.5.tar.gz
```bash
cd ahn-nlg-tools
mkdir -p angt/data/eval/3rdparty
mv mteval-v14c.pl meteor-1.5.tar.gz angt/data/eval/3rdparty/
tar -zxvf meteor-1.5.tar.gz
```

### Evaluation for Groundness 
- to use evaluation metrics for groundenss defined in [**CBR task**](https://arxiv.org/pdf/1906.02738.pdf)
- The result is different from the result of [**CBR task**](https://arxiv.org/pdf/1906.02738.pdf)

| Script                         | Precision | Recall | F1    |
|--------------------------------|-----------|--------|-------|
| human #1 reported in CbR task  | 2.89%     | 0.45%  | 0.78% |
| human #1 using our script       | 2.87%     | 0.44%  | 0.77% |
 - I cannot find the cause of this difference yet. 
