#include "windmill.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;
using namespace cv;

// 定义用于比较轮廓面积的比较函数
auto contourAreaComp = [](const vector<Point>& a, const vector<Point>& b) {
            return contourArea(a) < contourArea(b);
        };

// 定义残差函数
struct CosResidual {
    CosResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const parameters, T* residual) const {
        T A=parameters[0];
        T w=parameters[1];
        T phi=parameters[2];
        T b=parameters[3];    
    residual[0] = T(y_)-sin(b*x_+A/w*(cos(phi+T(1.57))-cos(w*x_+phi+T(1.57))));
    return true;
}
    private:
        const double x_;
        const double y_;
};


// 计算夹角角度正弦
double cal(Point2f x, Point2f y){
    double area=0;
    Point2f yuan;
    yuan.x=0;
    yuan.y=0;
    double r1=norm(x-yuan),r2=norm(y-yuan),r3=norm(x-y);
    double p=(r1+r2+r3)/2;
    area=sqrt(p*(p-r1)*(p-r2)*(p-r3));
    return 2*area/(r1*r2);
}


// 真实值
double A_true =0.785;
double w_true =1.884;
double phi_true=0.24;
double b_true=1.305;

// phi只要收敛即可
double dphi;
double pphi = 0;

// 判断是否收敛,误差在5%以内
bool has_converged(double * parameters) {
    if(fabs(parameters[0]) > A_true*0.95 && fabs(parameters[0]) < A_true*1.05 
        && fabs(parameters[1]) > w_true*0.95 && fabs(parameters[1]) < w_true * 1.05
        && fabs(parameters[3]) > b_true*0.95 && fabs(parameters[3]) < b_true * 1.05
        && dphi < 0.05) return 1; //phi只要收敛即可
    else return 0;
}

int main()
{
    double t_sum=0;
    for(int i=1;i<=10;i++){         //运行10次取均值
        cout << "Outer Iteration: " <<i << endl; 
        std::chrono::milliseconds t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        double t_start = (double)t.count();
        WINDMILL::WindMill wm(t.count());
        cv::Mat src;

        // 初始参数
        double b = 2.305, A = 1.785, w = 2.884, phi = 1.24;
        double parameters[4] = {A, w, phi, b};

        //计数器
        int count = -1;

        double pt = t_start ,dt;

        std::vector<double> x_data;
        std::vector<double> y_data;

        // starttime
        int64 start_time = getTickCount();

        // 记录初始目标点位置
        Point2f R_start,center_start, opp_start;

        //构建问题
        ceres::Problem problem;
        problem.AddParameterBlock(parameters, 4);
        

        while (1){
            if(count > 100){ 
                int64 end_time = getTickCount();
                t_sum +=( (end_time - start_time) / getTickFrequency());
                i--;
                break;
            }
            count++;
            cout << "Inner Iteration: " << count << endl;  // 输出当前迭代次数
            cout << "Outer Iteration: " << i << endl;  // 输出外层循环的当前迭代次数
            t = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch());
            src = wm.getMat((double)t.count());
            double t_now = (double)t.count();
            dt = (t_now - pt)/1000;
            if(count!=0) x_data.push_back((t_now-t_start)/1000.0);
            pt = t_now;
 
            
            
            
            //二值化图像
            cvtColor(src,src,COLOR_BGR2GRAY);
            threshold(src,src,30,255,THRESH_BINARY);
            
            
            //寻找轮廓
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(src, contours, hierarchy,RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


            // //找到R与锤子的面积-------锤子在3000-5000之间，R在100-300之间，其余的都是10000以上
            // for (size_t i = 0; i < contours.size(); i++) {
            // // 计算轮廓面积
            //     double area = contourArea(contours[i]);
            //     cout<<i<<":"<<area<<endl;
            // }          
            

            // 找到 r_con以及ham_con
            auto r_con_it = min_element(contours.begin(), contours.end(), contourAreaComp);
            vector<Point> r_con= *r_con_it;
            contours.erase(r_con_it);
            auto ham_con_it = std::min_element(contours.begin(), contours.end(), contourAreaComp);        
            vector<Point> ham_con= *ham_con_it;
            

            // 找到r和锤子的最小外接圆坐标和半径
            Mat src_cl=src.clone();
            Point2f center_r,center_ham;
            float radius_r,radius_ham;
            minEnclosingCircle(r_con, center_r, radius_r);
            minEnclosingCircle(ham_con, center_ham, radius_ham);
            // circle(src_cl,center_r,radius_r,Scalar(255,255,255),2);
            // circle(src_cl,center_ham,radius_ham,Scalar(255,255,255),2);
            // imwrite("../image/recognization.png",src_cl);
            // imshow("src",src_cl);

            // // 创建掩码并绘制外轮廓
            // Mat mask = Mat::zeros(src.size(), CV_8UC1);
            // drawContours(mask, contours, -1, Scalar(255), FILLED);
            // // 在掩码图像中查找内轮廓
            // vector<vector<Point>> innerContours;
            // vector<Vec4i> innerHierarchy;
            // findContours(mask, innerContours, innerHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);                
            // Point2f center_squ;
            // float radius_squ;
            // minEnclosingCircle(innerContours[0], center, radius);
            // circle(src, center, 5, Scalar(255, 255, 255), FILLED); // 绘制圆心
            
            // // 绘制r与锤子之间的圆形
            // float distance_1=norm(center_r-center_ham);
            // float radius=(distance_1-radius_ham-radius_r)/2;
            // float distance_2=radius_r+radius;
            // Point2f center;
            // center.x=center_r.x+(distance_2/distance_1)*(center_ham.x-center_r.x);
            // center.y=center_r.y+(distance_2/distance_1)*(center_ham.y-center_r.y);
            // circle(src,center,radius,Scalar(255,255,255),2);
            
            if(count==0){
                R_start=center_r;
                center_start=center;
                opp_start.x=center_start.x-R_start.x;
                opp_start.y=center_start.y-R_start.y;                
            }
            Point2f opp;
            opp.x=center.x-center_r.x;
            opp.y=center.y-center_r.y;            
            if(count!=0) y_data.push_back(cal(opp_start,opp));


            imshow("windmill",src);
            if (waitKey(1)>=0){break;}

                 for (int j = 0; j < x_data.size(); ++j) {
                    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CosResidual, 1, 4>(new CosResidual(x_data[j], y_data[j]));
                    problem.AddResidualBlock(cost_function,nullptr, parameters);
                }
                

                //配置求解器
                ceres::Solver::Options options;
                options.max_num_iterations = 1000;
                options.linear_solver_type = ceres::DENSE_QR;
                

                problem.SetParameterLowerBound(parameters, 3, 1);
                problem.SetParameterUpperBound(parameters, 3, 1.5);
                problem.SetParameterLowerBound(parameters, 0, 0.5);
                problem.SetParameterUpperBound(parameters, 0, 1);
                problem.SetParameterLowerBound(parameters, 1, 1.5);
                problem.SetParameterUpperBound(parameters, 1, 2);
                problem.SetParameterLowerBound(parameters, 2, 0.2);
                problem.SetParameterUpperBound(parameters, 2, 0.5);
               
                 //运行求解器
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);


                cout << "A0: " << parameters[0] << ", true_A0: " << b_true << endl;
                cout << "A: " << parameters[1] << ", true_A: " << A_true << endl;
                cout << "omega: " << parameters[2] << ", true_omega: " << w_true << endl;
                cout << "phi: " << parameters[3] << ", true_phi: " << phi_true << endl; 

                dphi = parameters[2] - pphi;
                pphi = parameters[2];
                // 判断
                if(has_converged(parameters)){
                    int64 end_time = getTickCount();
                    t_sum += (end_time - start_time) / getTickFrequency();
                    break;
                }
                else{}
            }

            

            

            //=======================================================//
            // imshow("windmill", src);
            // waitKey(1);
        }

    
    cout << "Average time: " << t_sum/10 << "seconds" << endl;
  
    return 0;
}



    // 创建掩码并绘制外轮廓
    Mat mask = Mat::zeros(src.size(), CV_8UC1);
    drawContours(mask, contours, -1, Scalar(255), FILLED); // 绘制外轮廓

    // 在掩码图像中查找内轮廓
    vector<vector<Point>> innerContours;
    vector<Vec4i> innerHierarchy;
    findContours(mask, innerContours, innerHierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
          
    Point2f center_squ;
    float radius_squ;
    minEnclosingCircle(innerContours[0], center, radius);
    circle(src, center, 5, Scalar(255, 255, 255), FILLED); // 绘制圆心

        // 显示结果
        imshow("Enclosing Circle", result);
        waitKey(0);
    }

    return 0;
}