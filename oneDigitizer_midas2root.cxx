// Example Program for converting MIDAS format to ROOT format.
//
// T. Lindner (Jan 2016) 
//
// Example is for the CAEN V792 ADC module

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <vector>

#include "TRootanaEventLoop.hxx"
#include "TFile.h"
#include "TTree.h"

//#include "TAnaManager.hxx"
#include "TV1720RawData.h"
#include "TDT743RawData.hxx"

#include "TH1D.h"

#ifdef USE_V792
#include "TV792Data.hxx"
#endif

class Analyzer: public TRootanaEventLoop {

public:

  // An analysis manager.  Define and fill histograms in 
  // analysis manager.
  //TAnaManager *anaManager;

  // The tree to fill.
  TTree *fTree1;

  int timestamp;
  uint64_t digiCounter1;
  uint64_t digiCounter2;
  int serialnumber;
  int midasEvent1;
  int midasEvent2;
  int frequency;
  int waveform_array[4][1536];
  // CAEN DT5720 tree variables

  TH1D *waveforms[4];


  Analyzer() {

  };

  virtual ~Analyzer() {};

  void Initialize(){


  }
  
  
  void BeginRun(int transition,int run,int time){

    std::cout << "Create histograms" << std::endl;	  
    for(int i=0; i<4; i++) waveforms[i] = new TH1D(Form("waveform%d",i),Form("waveform%d",i),1536,0,1536);	  
    
    // Create a TTree
    fTree1 = new TTree("midas_data1","First Digitizer");

    fTree1->Branch("midasEvent",&midasEvent1,"midasEvent/I");
    fTree1->Branch("timestamp",&timestamp,"timestamp/I");
    fTree1->Branch("serialnumber",&serialnumber,"serialnumber/I");
    fTree1->Branch("freqsetting",&frequency,"freqsetting/I");
    fTree1->Branch("triggerTime",&digiCounter1,"triggerTime/g");
    std::cout << "Set Branches for histograms" << std::endl;	  
    for(int i=0; i<4; i++) {
	    	fTree1->Branch(Form("Channel%d",i),"TH1D",&(waveforms[i]));
	    	fTree1->Branch(Form("Channel%d_arr",i),&(waveform_array[i]), Form("waveformArray%d[1536]/I,",i));

    }



    midasEvent1 = -100;
    midasEvent2 = -100;

    std::cout << "Completed Begin Run" << std::endl;

  }   


  void EndRun(int transition,int run,int time){
	std::cout << fTree1->GetEntries() << std::endl;
        printf("\n");
  }

  
  
  // Main work here; create ttree events for every sequenced event in 
  // Lecroy data packets.
  bool ProcessMidasEvent(TDataContainer& dataContainer){

    serialnumber = dataContainer.GetMidasEvent().GetSerialNumber();
    if(serialnumber%10 == 0) printf(".");
    timestamp = (time_t)dataContainer.GetMidasEvent().GetTimeStamp();
    std::cout << "Process Event " <<  serialnumber << std::endl;
 
#ifdef USE_V792    
    TV792Data *data = dataContainer.GetEventData<TV792Data>("ADC0");
    if(data){
      nchannels = 32;
      for(int i = 0; i < nchannels;i++) adc_value[i] = 0;
      
      /// Get the Vector of ADC Measurements.
      std::vector<VADCMeasurement> measurements = data->GetMeasurements();
      for(unsigned int i = 0; i < measurements.size(); i++){ // loop over measurements
        
        int chan = measurements[i].GetChannel();
        uint32_t adc = measurements[i].GetMeasurement();
        
        if(chan >= 0 && chan < nchannels)
          adc_value[chan] = adc;
      }
    }
#endif

    std::cout << "Get the raw data" << std::endl;
    //Fill the ttrees
    TDT743RawData *dt743 = dataContainer.GetEventData<TDT743RawData>("D720");
    std::cout << dt743 << std::endl;
    TDT743RawData *dt743_mod2 = dataContainer.GetEventData<TDT743RawData>("43F2");
    if(dt743){

      //std::cout << "Has first digitizer" << std::endl;	    
      midasEvent1 = serialnumber;

      std::vector<RawChannelMeasurement> measurements = dt743->GetMeasurements();
      std::cout << "MEASUREMENTS SIZE: " << measurements.size() << std::endl;

      digiCounter1 = (uint64_t)dt743->GetTriggerTimeTag(); 
      //std::cout << dt743->GetTriggerTimeTag() << std::endl;
      //std::cout << "Reset histograms" << std::endl;
      //for(int i=0; i<8; i++) waveforms[i]->Reset();

      std::cout << "measurement size!: " << measurements.size() << std::endl;
      std::cout << "digi Counter 1: " << digiCounter1 << std::endl;

      for(int i = 0; i < measurements.size(); i++){

        int chan = measurements[i].GetChannel();
        std::cout << "Measurement " << i << " Channel " << chan << std::endl;
        int nsamples = measurements[i].GetNSamples();
        //if(nsamples!=1536) std::cout << "Number of samples isn't 1536" << std::endl;
        if(i%2==0) frequency = measurements[i].GetFrequency();
	double period = 1.0/3.2;
	if(frequency==1) period = 1.0/1.6;
	else if(frequency==2) period = 1.0/0.8;
	else if(frequency==3) period = 1.0/0.4;

        if(waveforms[i]!=NULL) waveforms[i]->Delete();
	waveforms[i] = new TH1D(Form("waveform%d",i),Form("waveform%d",i),nsamples,0,(float)nsamples*period); 

	std::cout << "NSAMPLES: " << nsamples << std::endl;
        for(int ib = 0; ib < nsamples; ib++){
          waveforms[i]->SetBinContent(ib+1, measurements[i].GetSample(ib));
	  waveform_array[i][ib] = measurements[i].GetSample(ib);
	  //waveform_array[i].push_back(measurements[i].GetSample(ib));
        }
      }
      fTree1->Fill();
      //waveform_array->clear();
    }    


    if(dt743_mod2){

      //std::cout << "Has second digitizer" << std::endl;	    
      midasEvent2 = serialnumber;

      std::vector<RawChannelMeasurement> measurements = dt743_mod2->GetMeasurements();

      digiCounter2 = (uint64_t)dt743_mod2->GetTriggerTimeTag(); 
      //std::cout << "Reset histograms" << std::endl;
      //for(int i=0; i<8; i++) waveforms[i]->Reset();


      for(int i = 0; i < measurements.size(); i++){

        int chan = measurements[i].GetChannel();
        //std::cout << "Measurement " << i << " Channel " << chan << std::endl;
        int nsamples = measurements[i].GetNSamples();
        //std::cout << "NSAMPLES: " << nsamples << std::endl;
        //if(nsamples!=1536) std::cout << "Number of samples isn't 1536" << std::endl;
        if(i%2==0) frequency = measurements[i].GetFrequency();
	double period = 1.0/3.2;
	if(frequency==1) period = 1.0/1.6;
	else if(frequency==2) period = 1.0/0.8;
	else if(frequency==3) period = 1.0/0.4;

        if(waveforms[i+8]!=NULL) waveforms[i+8]->Delete();
	waveforms[i+8] = new TH1D(Form("waveform%d",i+8),Form("waveform%d",i+8),nsamples,0,(float)nsamples*period); 

        for(int ib = 0; ib < nsamples; ib++){
          waveforms[i+8]->SetBinContent(ib+1, measurements[i].GetSample(ib));
        }
      }
    }    



    return true;

  };
  
  // Complicated method to set correct filename when dealing with subruns.
  std::string SetFullOutputFileName(int run, std::string midasFilename)
  {
    char buff[128]; 
    Int_t in_num = 0, part = 0;
    Int_t num[2] = { 0, 0 }; // run and subrun values
    // get run/subrun numbers from file name
    for (int i=0; ; ++i) {
      char ch = midasFilename[i];
        if (!ch) break;
        if (ch == '/') {
          // skip numbers in the directory name
          num[0] = num[1] = in_num = part = 0;
        } else if (ch >= '0' && ch <= '9' && part < 2) {
          num[part] = num[part] * 10 + (ch - '0');
          in_num = 1;
        } else if (in_num) {
          in_num = 0;
          ++part;
        }
    }
    if (part == 2) {
      if (run != num[0]) {
        std::cerr << "File name run number (" << num[0]
                  << ") disagrees with MIDAS run (" << run << ")" << std::endl;
        exit(1);
      }
      sprintf(buff,"/home/t2k_otr/data_root/root_run_%.6d_%.4d.root", run, num[1]);
      printf("Using filename %s\n",buff);
    } else {
      sprintf(buff,"/home/t2k_otr/data_root/root_run_%.6d.root", run);
    }
    return std::string(buff);
  };





}; 


int main(int argc, char *argv[])
{

  Analyzer::CreateSingleton<Analyzer>();
  return Analyzer::Get().ExecuteLoop(argc, argv);

}

