#include "ecocar_fusion/gaussMapRos.hpp"
#include "ecocar_fusion/utils.hpp"

#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseArray.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "jsk_recognition_msgs/BoundingBoxArray.h"

#include "autoware_msgs/DetectedObjectArray.h"
#include "autoware_msgs/DetectedObject.h"

GaussMapRos::GaussMapRos(ros::NodeHandle nh, ros::NodeHandle nh_p): nh_(nh), nh_p_(nh_p){
    ROS_INFO_STREAM("[ecocar_fusion_node] Initializing GaussMap...");


    int mapWidth;               // Gauss map width
    int mapHeight;              // Gauss map height
    int mapResolution;          // Gauss map resolution (meter per pixel?)
    radarDistri = (distInfo_t*)malloc(sizeof(struct DistributionInfo));

    // Get ROS parameters
    nh_p_.getParam("mobileye_topic", mobileyeTopic_);
    nh_p_.getParam("bosch_topic", radarTopic_);
    nh_p_.getParam("bbox_vis_topic", bboxVisTopic_);
    nh_p_.getParam("bbox_topic", bboxTopic_);
    nh_p_.getParam("map_width", mapWidth);
    nh_p_.getParam("map_height", mapHeight);
    nh_p_.getParam("map_resolution", mapResolution);
    nh_p_.getParam("stdDev", radarDistri->stdDev);
    nh_p_.getParam("mean", radarDistri->mean);
    nh_p_.getParam("radius_cutoff", radarDistri->distCutoff);
    nh_p_.getParam("update_rate", _refreshRate);
    nh_p_.getParam("visualize_boxes", visualizeBoxes_);
    nh_p_.getParam("radar_max_age", _radMaxAge);
    nh_p_.getParam("mobileye_max_age", _meMaxAge);
    nh_p_.getParam("min_exist_prob", minExistProb);
    nh_p_.getParam("assoc_adjust", adjustFactor);
    GaussMap::init(mapHeight,mapWidth, mapResolution, useMin);

    // setup the diagnostics updater
    updater_.setHardwareID("GaussMap");
    updater_.broadcast(0, "initializing");

    // initialize the tf listener
    tfListener = new tf2_ros::TransformListener(tfBuffer);

    // create a subscriber for the mobileye detections topic
    ROS_INFO_STREAM("[ecocar_fusion_node] Subscribing to " << mobileyeTopic_);
    meSub = nh_.subscribe(mobileyeTopic_, 100, &GaussMapRos::mobileyeCallback, this);
    
    // create a subscriber for the radar detections topic
    ROS_INFO_STREAM("[ecocar_fusion_node] Subscribing to " << radarTopic_);
    radSub = nh_.subscribe(radarTopic_, 100, &GaussMapRos::radarCallback, this);

    if(visualizeBoxes_){
        // create a publisher for the fusion results
        ROS_INFO_STREAM("[ecocar_fusion_node] Publishing visualizable boxes to " << bboxVisTopic_);
        pub_bbox_vis_ = nh_.advertise<jsk_recognition_msgs::BoundingBoxArray>(bboxVisTopic_, 100);
        ROS_INFO_STREAM("[ecocar_fusion_node] Publishing visualizable velocities to " << bboxVisTopic_ << "/pose");
        pub_pose_ = nh_.advertise<geometry_msgs::PoseArray>(bboxVisTopic_ + "/pose", 100);
        grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/fused/occupancyGrid", 100);
    }
    ROS_INFO_STREAM("[ecocar_fusion_node] Publishing detected objects to " << bboxTopic_);
    pub_bbox_ = nh_.advertise<autoware_msgs::DetectedObjectArray>(bboxTopic_,100);

    ROS_INFO("[ecocar_fusion_node] Using window size: %d (%lu)", winSize, searchSize);
    pub_bbox_freq_ = new diagnostic_updater::DiagnosedPublisher<autoware_msgs::DetectedObjectArray>(
        pub_bbox_, 
        updater_, 
        diagnostic_updater::FrequencyStatusParam(&_refreshRate, &_refreshRate, .1, 10),
        diagnostic_updater::TimeStampStatusParam(0.0, 0.05)      // min and max publishing delay
    );


    // create a timer for fusing the radar and camera data
    ros::TimerOptions opts;
    timer_ = nh_.createTimer(ros::Duration(1.0/_refreshRate), &GaussMapRos::runFusion, this);

    ROS_INFO_STREAM("[ecocar_fusion_node] Initialization complete!");
    updater_.broadcast(0, "initialization complete");
}

GaussMapRos::~GaussMapRos(){
    delete tfListener;
    delete pub_bbox_freq_;
}

void GaussMapRos::radarCallback(const sensor_msgs::PointCloud2 &msg){
    // puts radar data from the radar pointcloud into a vector in the form of:
    // [x,y,vx,vy, x,y,vx,vy, x,y,vx,vy, ...]
    ros::Time received = ros::Time::now();
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(msg,          "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(msg,          "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_vx(msg,         "vx");
    sensor_msgs::PointCloud2ConstIterator<float> iter_vy(msg,         "vy");
    sensor_msgs::PointCloud2ConstIterator<float> iter_wExist(msg,     "wExist");
    sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_targetId(msg, "targetId");
    sensor_msgs::PointCloud2ConstIterator<float> iter_xSig(msg,       "dxSigma");
    sensor_msgs::PointCloud2ConstIterator<float> iter_ySig(msg,       "dySigma");

    std::vector<float> curData;
    curData.reserve(msg.width*4);
    while(iter_x != iter_x.end()){
        if(iter_wExist[0] > minExistProb) {
            curData.push_back(iter_x[0]);    //x
            curData.push_back(iter_y[0]);    //y
            curData.push_back(iter_vx[0]);   //vx
            curData.push_back(iter_vy[0]);   //vy
            curData.push_back(iter_wExist[0]);
            curData.push_back(iter_targetId[0]);
            curData.push_back(iter_xSig[0]);
            curData.push_back(iter_ySig[0]);
        }
        ++iter_x;
        ++iter_y;
        ++iter_vx;
        ++iter_vy;
        ++iter_wExist;
        ++iter_targetId;
        ++iter_xSig;
        ++iter_ySig;
    }

    radMapMutex.lock();
    _radData.insert(std::make_pair(received, curData));
    radMapMutex.unlock();
}

void GaussMapRos::cleanRadMap(){
    ros::Time now = ros::Time::now();
    ros::Duration diff;
    
    std::map<ros::Time, std::vector<float> >::iterator tmp;
    for(std::map<ros::Time, std::vector<float> >::iterator it = _radData.begin(); \
        it != _radData.end();)
    {
        diff = now - it->first;
        if(diff.nsec > (_radMaxAge *1000000)){
            tmp = it;
            it++;
            _radData.erase(tmp);
        }else
            it++;
    }
}

// this template py:array_t forces the numpy array to be passed without any strides
// and favors a c-style array
void GaussMapRos::addRadarData(const float* array_in, array_info info){
    if(radarData != nullptr)
        throw std::runtime_error("addRadarData can only be called once after calling reset()");
    size_t radarPoints, radarFeatures;
    
    if(info.elementSize != sizeof(RadarData_t)){
        throw std::runtime_error("Invalid datatype passed with radar data. Should be type: float (float32).");
    }

    radarPoints = info.rows;            // num points
    if(radarPoints == 0) return;            // do nothing if there are no points;
    radarFeatures = info.cols;          // usually 18

    radarInfo.elementSize = sizeof(RadarData_t);
    radarInfo.cols = radarFeatures;
    radarInfo.rows = radarPoints;

    // allocate and copy the array to the GPU so we can run a kernel on it
    safeCudaMalloc(&radarData, sizeof(RadarData_t) * radarPoints * radarFeatures);
    safeCudaMemcpy2Device(radarData, array_in, sizeof(RadarData_t) * radarPoints * radarFeatures);

    calcRadarMap();     // setup for the CUDA kernel. in GPU code
}

array_t GaussMapRos::getRadarData(){
    radMapMutex.lock();
    cleanRadMap();

    // get the number of total radar points from all three radars
    std::map<ros::Time,std::vector<float> >::const_iterator it;
    size_t elements = 0;
    for(it = _radData.begin(); it != _radData.end(); it++){
        elements += it->second.size();
    }

    // store all of the elements we need in radar
    float* data;
    data = (float*)malloc(sizeof(float) * elements);
    size_t ptr = 0;
    for(it = _radData.begin(); it != _radData.end(); it++){
        memcpy(data+ptr, it->second.data(), it->second.size()* sizeof(float));
        ptr += it->second.size();
    }
    radMapMutex.unlock();

    array_info info;
    info.cols = 8;  // [x,y,vx,vy,wExist,targetId,xSigma,ySigma]
    info.elementSize = sizeof(float);
    info.rows = elements/8;

    array_t ret;
    ret.data = data;
    ret.info = info;
    return ret;
}

void GaussMapRos::addCameraData(const float* array_in, array_info info){

    if(info.elementSize != sizeof(float)){
        throw std::runtime_error("Invalid datatype passed with camera data. Expected float32");
    }

    camInfo.cols = info.cols;
    camInfo.rows = info.rows;
    camInfo.elementSize = info.elementSize;

    safeCudaMalloc(&camData, camInfo.size());
    safeCudaMemcpy2Device(camData, array_in, camInfo.size());

    safeCudaMemcpy2Device(camInfo_cuda, &camInfo, sizeof(array_info));

}

void GaussMapRos::mobileyeCallback(const mobileye_560_660_msgs::ObstacleData &msg){
    obstacle_t curObstacle;
    
    
    meMapMutex.lock();
    std::map<uint16_t,obstacle_t>::iterator it;
    it = _meData.find(msg.obstacle_id);
    
    if(it != _meData.end())
        curObstacle = it->second;
    else
        curObstacle = new struct objStruct;

    curObstacle->object[0] = msg.obstacle_pos_x;
    curObstacle->object[1] = msg.obstacle_pos_y;
    curObstacle->object[2] = msg.obstacle_type+1;
    curObstacle->object[3] = msg.obstacle_width;
    curObstacle->time = ros::Time::now();
    curObstacle->frame_id = msg.header.frame_id;

    try{
        geometry_msgs::TransformStamped frameTf = tfBuffer.lookupTransform("vehicle", msg.header.frame_id, ros::Time(0));
        struct objStruct tmp;
        tmp = tfBuffer.transform(*curObstacle, "vehicle");
        *curObstacle = tmp;
    }catch(tf2::TransformException &ex){
        ROS_WARN_THROTTLE(1, "%sNot transforming", ex.what());
    }

    _meData.insert(std::make_pair(msg.obstacle_id, curObstacle));
    meMapMutex.unlock();
}

array_t GaussMapRos::getMobileyeData(){
    // stored as [x,y,class,width,...]
    meMapMutex.lock();
    cleanMeMap();
    std::vector<float> retVec;
    retVec.reserve(4*_meData.size());
    
    for(std::map<uint16_t,obstacle_t>::iterator it = _meData.begin(); \
        it != _meData.end(); it++)
    {
        for(size_t i = 0; i < 4; i++)
            retVec.push_back(it->second->object[i]);
    }
    meMapMutex.unlock();
    
    // put the data into a shared pointer so we don't have to worry about
    // deleting it later
    float* data;
    data = (float*)malloc(retVec.size() * sizeof(float));
    memcpy(data, retVec.data(), sizeof(float) * retVec.size());

    // setup the size so it's not lost
    array_info info;
    info.cols = 4;
    info.rows = retVec.size() / 4;
    info.elementSize = sizeof(float);

    array_t ret;
    ret.info = info;
    ret.data = data;

    return ret;
}

void GaussMapRos::cleanMeMap(){
    // remove elements from the map that are too old
    // i.e. older than _maxAge ms from NOW
    ros::Time now = ros::Time::now();
    ros::Duration diff;
  
    std::map<uint16_t,obstacle_t>::iterator tmp;
    for(std::map<uint16_t,obstacle_t>::iterator it = _meData.begin(); \
        it != _meData.end();)
    {
        diff = now - it->second->time;
        if(diff.nsec > (_meMaxAge *1000000)){
            tmp = it;
            it++;
            _meData.erase(tmp);
        }else
            it++;
    }
}

void GaussMapRos::runFusion(const ros::TimerEvent &event){
    // use a mutex to protect against running again before the last run has returned
    if(!runLock.try_lock()){
        ROS_WARN_STREAM("Skipping fusion run. Timer event called during previous run");
        return;
    }
    
    array_t camData = getMobileyeData();
    array_t radData = getRadarData();
    if(camData.info.rows == 0 && radData.info.rows == 0){
        free(camData.data);
        free(radData.data);
        runLock.unlock();
        return;
    }
    
    if(radData.info.rows == 0){
        ROS_WARN_STREAM("[gaussMap] no radar data this fusion cycle");
        float* ret = (float*)malloc(camData.info.rows * sizeof(float) * 5);
        array_info nfo;
        nfo.elementSize = sizeof(float);
        nfo.cols = 5;
        nfo.rows = camData.info.rows;

        for(size_t i = 0; i < camData.info.rows; i++){
            ret[i*5 + 0] = camData.data[i * camData.info.cols + 0];
            ret[i*5 + 1] = camData.data[i * camData.info.cols + 1];
            ret[i*5 + 2] = 0;
            ret[i*5 + 3] = 0;
            ret[i*5 + 4] = camData.data[i * camData.info.cols + 2]; // class
        }

        array_t toSend;
        toSend.info = nfo;
        toSend.data = ret;
        sendObjs(toSend, event.current_real);
        free(ret);
        
        free(camData.data);
        free(radData.data);
        runLock.unlock();
        return;
    }

    addRadarData(radData.data, radData.info);
    addCameraData(camData.data, camData.info);

    // perform the actual fusion
    std::pair<array_info,float*> fused;
    fused = associatePair();

    array_t toSendFused;
    toSendFused.data = fused.second;
    toSendFused.info = fused.first;
    sendObjs(toSendFused, event.current_real);

    reset();
    free(fused.second);
    free(camData.data);
    free(radData.data);
    updater_.update();
    runLock.unlock();
}

void GaussMapRos::sendObjs(array_t fused, ros::Time start){
    // fused.data: [x,y,vx,vy,class]

    jsk_recognition_msgs::BoundingBoxArray bboxVisArry;
    geometry_msgs::PoseArray poseArry;
    poseArry.header.frame_id = "vehicle";
    bboxVisArry.header.frame_id = "vehicle";
    float alpha;

    autoware_msgs::DetectedObjectArray bboxArray;
    bboxArray.header.frame_id = "vehicle";
    bboxArray.header.stamp = start;
/*    for(size_t i = 0; i < fused.info.rows; i++){
        putchar('[');
        for(size_t j = 0; j < fused.info.cols; j++){
            printf("%f,", fused.data[array_index_cpu(i, j, &fused.info)]);
        }
        printf("]\n");
    }
    printf("---\n");
*/
    for(size_t obj = 0; obj < fused.info.rows; obj++){
        geometry_msgs::Pose pose;
        geometry_msgs::Pose orientation;
        geometry_msgs::Vector3 dimension;
        geometry_msgs::Twist velocity;

        autoware_msgs::DetectedObject bbox;

        pose.position.x = fused.data[fused.info.cols * obj + 0];
        pose.position.y = fused.data[fused.info.cols * obj + 1];
        pose.position.z = 1;

        pose.orientation.w = 0;
        pose.orientation.x = 0;
        pose.orientation.y = 0;
        pose.orientation.z = 1;

        dimension.x = 1;
        dimension.y = 1;
        dimension.z = 1;

        velocity.linear.x = fused.data[fused.info.cols * obj + 2];
        velocity.linear.y = fused.data[fused.info.cols * obj + 3];
        velocity.linear.z = 0;
        velocity.angular.x = 0;
        velocity.angular.y = 0;
        velocity.angular.z = 0;

        bbox.space_frame = "vehicle";
        bbox.valid = true;
        bbox.header.frame_id = "vehicle";
        bbox.pose = pose;
        bbox.dimensions = dimension;
        bbox.velocity = velocity;
        bbox.pose_reliable = true;
        bbox.velocity_reliable = true;
        bbox.acceleration_reliable = true;
        bbox.id = obj;

        bbox.label = std::to_string((int)fused.data[fused.info.cols * obj + 4]);

        bboxArray.objects.push_back(bbox);


        if(visualizeBoxes_){
            jsk_recognition_msgs::BoundingBox bboxVis;
            geometry_msgs::Pose vel;
            bboxVis.header.frame_id = "vehicle";
            bboxVis.dimensions = dimension;

            bboxVis.pose.position = pose.position;

            // identity quaternion
            bboxVis.pose.orientation.w = 0;
            bboxVis.pose.orientation.x = 0;
            bboxVis.pose.orientation.y = 0;
            bboxVis.pose.orientation.z = 1;

            
            bboxVis.value = obj;
            bboxVis.label = fused.data[fused.info.cols * obj + 4];
            bboxVisArry.boxes.push_back(bboxVis);

            if(fused.data[fused.info.cols * obj + 3] != 0.0)
                alpha = sin(fused.data[fused.info.cols * obj + 2] / fused.data[fused.info.cols * obj + 3]);
            else
                alpha = 0;
        
            vel.orientation.w = isnan(alpha) || isinf(alpha) ? 0 : cos(alpha/2);
            vel.orientation.x = 0;
            vel.orientation.y = 0;
            vel.orientation.z = isnan(alpha) || isinf(alpha) ? 0 : sin(alpha/2);
            vel.position = pose.position;

            poseArry.poses.push_back(vel);
        }
    }
    if(visualizeBoxes_){
        pub_bbox_vis_.publish(bboxVisArry);
        pub_pose_.publish(poseArry);
        sendGrid();
    }
    pub_bbox_freq_->publish(bboxArray);
    // pub_bbox_freq_->tick();
}

void GaussMapRos::sendGrid(){
    nav_msgs::OccupancyGrid toPub;
    toPub.header.frame_id = "vehicle";
    toPub.header.stamp = ros::Time::now();
    toPub.info.map_load_time = ros::Time::now();
    toPub.info.resolution = 1.0/mapRel.res;
    toPub.info.width = mapInfo.cols;
    toPub.info.height = mapInfo.rows;

    toPub.info.origin.orientation.w = 1;
    toPub.info.origin.orientation.x = 0;
    toPub.info.origin.orientation.y = 0;
    toPub.info.origin.orientation.z = 0;
    toPub.info.origin.position.x = mapRel.height/-2.0;
    toPub.info.origin.position.y = mapRel.width/-2.0;

    mapType_t *tmp;
    tmp = (mapType_t*)malloc(mapInfo.cols * mapInfo.rows * sizeof(mapType_t));
    safeCudaMemcpy2Host(tmp, array, mapInfo.rows * mapInfo.cols * sizeof(mapType_t));

    for(int i = mapInfo.rows-1; i >= 0; i--){
        for(int j = 0; j < mapInfo.cols ; j++){
            toPub.data.push_back((int8_t)std::min(tmp[i*mapInfo.cols + j]*90, 100.0f));
        }
    }
    grid_pub_.publish(toPub);

    free(tmp);
}
