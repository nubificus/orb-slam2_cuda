/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

#include <iostream>

#include "PnPsolver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>

using namespace std;

namespace ORB_SLAM2
{


PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches):
    pws(0), us(0), alphas(0), pcs(0), maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0)
{
    mvpMapPointMatches = vpMapPointMatches;
    mvP2D.reserve(F.mvpMapPoints.size());
    mvSigma2.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());
    mvKeyPointIndices.reserve(F.mvpMapPoints.size());
    mvAllIndices.reserve(F.mvpMapPoints.size());

    int idx=0;
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if(pMP)
        {
            if(!pMP->isBad())
            {
                const cv::KeyPoint &kp = F.mvKeysUn[i];

                mvP2D.push_back(kp.pt);
                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);

                cv::Mat Pos = pMP->GetWorldPos();
                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));

                mvKeyPointIndices.push_back(i);
                mvAllIndices.push_back(idx);               

                idx++;
            }
        }
    }

    // Set camera calibration parameters
    fu = F.fx;
    fv = F.fy;
    uc = F.cx;
    vc = F.cy;

    SetRansacParameters();
}

PnPsolver::~PnPsolver()
{
  delete [] pws;
  delete [] us;
  delete [] alphas;
  delete [] pcs;
}


void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;
    mRansacMinSet = minSet;

    N = mvP2D.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    int nMinInliers = N*mRansacEpsilon;
    if(nMinInliers<mRansacMinInliers)
        nMinInliers=mRansacMinInliers;
    if(nMinInliers<minSet)
        nMinInliers=minSet;
    mRansacMinInliers = nMinInliers;

    if(mRansacEpsilon<(float)mRansacMinInliers/N)
        mRansacEpsilon=(float)mRansacMinInliers/N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if(mRansacMinInliers==N)
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(mRansacEpsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    mvMaxError.resize(mvSigma2.size());
    for(size_t i=0; i<mvSigma2.size(); i++)
        mvMaxError[i] = mvSigma2[i]*th2;
}

cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers,nInliers);    
}

cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers.clear();
    nInliers=0;

    set_maximum_number_of_correspondences(mRansacMinSet);

    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    vector<size_t> vAvailableIndices;

    int nCurrentIterations = 0;
    while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
    {
        nCurrentIterations++;
        mnIterations++;
        reset_correspondences();

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for(short i = 0; i < mRansacMinSet; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            int idx = vAvailableIndices[randi];

            add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);

            vAvailableIndices[idx] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // Compute camera pose
        compute_pose(mRi, mti);

        // Check inliers
        CheckInliers();

        if(mnInliersi>=mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            if(mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3,3,CV_64F,mRi);
                cv::Mat tcw(3,1,CV_64F,mti);
                Rcw.convertTo(Rcw,CV_32F);
                tcw.convertTo(tcw,CV_32F);
                mBestTcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
            }

            if(Refine())
            {
                nInliers = mnRefinedInliers;
                vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
                for(int i=0; i<N; i++)
                {
                    if(mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }
                return mRefinedTcw.clone();
            }

        }
    }

    if(mnIterations>=mRansacMaxIts)
    {
        bNoMore=true;
        if(mnBestInliers>=mRansacMinInliers)
        {
            nInliers=mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
            for(int i=0; i<N; i++)
            {
                if(mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }

    return cv::Mat();
}

bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for(size_t i=0; i<mvbBestInliers.size(); i++)
    {
        if(mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    set_maximum_number_of_correspondences(vIndices.size());

    reset_correspondences();

    for(size_t i=0; i<vIndices.size(); i++)
    {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
    }

    // Compute camera pose
    compute_pose(mRi, mti);

    // Check inliers
    CheckInliers();

    mnRefinedInliers =mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if(mnInliersi>mRansacMinInliers)
    {
        cv::Mat Rcw(3,3,CV_64F,mRi);
        cv::Mat tcw(3,1,CV_64F,mti);
        Rcw.convertTo(Rcw,CV_32F);
        tcw.convertTo(tcw,CV_32F);
        mRefinedTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
        return true;
    }

    return false;
}


void PnPsolver::CheckInliers()
{
    mnInliersi=0;

    for(int i=0; i<N; i++)
    {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];

        float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
        float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
        float invZc = 1/(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);

        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;

        float distX = P2D.x-ue;
        float distY = P2D.y-ve;

        float error2 = distX*distX+distY*distY;

        if(error2<mvMaxError[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
        {
            mvbInliersi[i]=false;
        }
    }
}


void PnPsolver::set_maximum_number_of_correspondences(int n)
{
  if (maximum_number_of_correspondences < n) {
    if (pws != 0) delete [] pws;
    if (us != 0) delete [] us;
    if (alphas != 0) delete [] alphas;
    if (pcs != 0) delete [] pcs;

    maximum_number_of_correspondences = n;
    pws = new double[3 * maximum_number_of_correspondences];
    us = new double[2 * maximum_number_of_correspondences];
    alphas = new double[4 * maximum_number_of_correspondences];
    pcs = new double[3 * maximum_number_of_correspondences];
  }
}

void PnPsolver::reset_correspondences(void)
{
  number_of_correspondences = 0;
}

void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
{
  pws[3 * number_of_correspondences    ] = X;
  pws[3 * number_of_correspondences + 1] = Y;
  pws[3 * number_of_correspondences + 2] = Z;

  us[2 * number_of_correspondences    ] = u;
  us[2 * number_of_correspondences + 1] = v;

  number_of_correspondences++;
}

void PnPsolver::choose_control_points()
{
    // Take C0 as the reference points centroid:
    cws[0][0] = cws[0][1] = cws[0][2] = 0;
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            cws[0][j] += pws[3 * i + j];

    for (int j = 0; j < 3; j++)
        cws[0][j] /= number_of_correspondences;

    // Create matrix of zero-mean 3D points
    cv::Mat PW0(number_of_correspondences, 3, CV_64F);
    for (int i = 0; i < number_of_correspondences; i++)
        for (int j = 0; j < 3; j++)
            PW0.at<double>(i, j) = pws[3 * i + j] - cws[0][j];

    // Compute covariance matrix: PW0^T * PW0
    cv::Mat PW0tPW0 = PW0.t() * PW0;

    // Perform SVD
    cv::Mat W, U, Vt;
    cv::SVD::compute(PW0tPW0, W, U, Vt);

    // Construct control points C1, C2, C3 based on principal components
    for (int i = 1; i < 4; i++) {
        double k = std::sqrt(W.at<double>(i - 1) / number_of_correspondences);
        for (int j = 0; j < 3; j++) {
            cws[i][j] = cws[0][j] + k * U.at<double>(j, i - 1);  // U is 3x3
        }
    }
}

void PnPsolver::compute_barycentric_coordinates()
{
    // Create matrices for control points difference and inverse
    cv::Mat CC(3, 3, CV_64F);
    cv::Mat CC_inv;

    // Fill CC with differences: cws[j] - cws[0], j = 1 to 3
    for (int i = 0; i < 3; i++) {
        for (int j = 1; j < 4; j++) {
            CC.at<double>(i, j - 1) = cws[j][i] - cws[0][i];
        }
    }

    // Invert CC using SVD decomposition
    cv::invert(CC, CC_inv, cv::DECOMP_SVD);

    // Compute barycentric coordinates (alphas)
    for (int i = 0; i < number_of_correspondences; i++) {
        double* pi = pws + 3 * i;
        double* a = alphas + 4 * i;

        // Calculate a[1], a[2], a[3]
        for (int j = 0; j < 3; j++) {
            a[1 + j] =
                CC_inv.at<double>(j, 0) * (pi[0] - cws[0][0]) +
                CC_inv.at<double>(j, 1) * (pi[1] - cws[0][1]) +
                CC_inv.at<double>(j, 2) * (pi[2] - cws[0][2]);
        }

        // Compute a[0] to ensure sum to 1
        a[0] = 1.0 - a[1] - a[2] - a[3];
    }
}


void PnPsolver::fill_M(cv::Mat* M, const int row, const double* as, const double u, const double v)
{
    // Fill two rows: row and row+1
    for (int i = 0; i < 4; i++) {
        // First row
        M->at<double>(row,     3 * i    ) = as[i] * fu;
        M->at<double>(row,     3 * i + 1) = 0.0;
        M->at<double>(row,     3 * i + 2) = as[i] * (uc - u);

        // Second row
        M->at<double>(row + 1, 3 * i    ) = 0.0;
        M->at<double>(row + 1, 3 * i + 1) = as[i] * fv;
        M->at<double>(row + 1, 3 * i + 2) = as[i] * (vc - v);
    }
}

void PnPsolver::compute_ccs(const double* betas, const double* ut)
{
    // Initialize ccs to zero
    for (int i = 0; i < 4; i++)
        std::fill(ccs[i], ccs[i] + 3, 0.0);

    // Accumulate contributions
    for (int i = 0; i < 4; i++) {
        const double* v = ut + 12 * (11 - i);
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 3; k++) {
                ccs[j][k] += betas[i] * v[3 * j + k];
            }
        }
    }
}


void PnPsolver::compute_pcs(void)
{
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = alphas + 4 * i;
    double * pc = pcs + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

double PnPsolver::compute_pose(double R[3][3], double t[3])
{
    choose_control_points();
    compute_barycentric_coordinates();

    cv::Mat M = cv::Mat::zeros(2 * number_of_correspondences, 12, CV_64F);

    for (int i = 0; i < number_of_correspondences; i++)
        fill_M(&M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

    cv::Mat MtM = M.t() * M;

    cv::Mat w, u, vt;
    cv::SVD::compute(MtM, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    double ut[12 * 12];
    memcpy(ut, vt.ptr<double>(), sizeof(double) * 12 * 12);  // copy Vt to ut

    double l_6x10[6 * 10], rho[6];
    cv::Mat L_6x10(6, 10, CV_64F, l_6x10);
    cv::Mat Rho(6, 1, CV_64F, rho);

    compute_L_6x10(ut, l_6x10);
    compute_rho(rho);

    double Betas[4][4], rep_errors[4];
    double Rs[4][3][3], ts[4][3];

    find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
    gauss_newton(&L_6x10, &Rho, Betas[1]);
    rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

    find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
    gauss_newton(&L_6x10, &Rho, Betas[2]);
    rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

    find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
    gauss_newton(&L_6x10, &Rho, Betas[3]);
    rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

    int N = 1;
    if (rep_errors[2] < rep_errors[1]) N = 2;
    if (rep_errors[3] < rep_errors[N]) N = 3;

    copy_R_and_t(Rs[N], ts[N], R, t);

    return rep_errors[N];
}


void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
			double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

double PnPsolver::dist2(const double * p1, const double * p2)
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPsolver::dot(const double * v1, const double * v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
{
  double sum2 = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    double * pw = pws + 3 * i;
    double Xc = dot(R[0], pw) + t[0];
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
    double ue = uc + fu * Xc * inv_Zc;
    double ve = vc + fv * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];

    sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
  }

  return sum2 / number_of_correspondences;
}

void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
{
    double pc0[3] = {0.0, 0.0, 0.0};
    double pw0[3] = {0.0, 0.0, 0.0};

    for (int i = 0; i < number_of_correspondences; i++) {
        const double* pc = pcs + 3 * i;
        const double* pw = pws + 3 * i;
        for (int j = 0; j < 3; j++) {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        }
    }

    for (int j = 0; j < 3; j++) {
        pc0[j] /= number_of_correspondences;
        pw0[j] /= number_of_correspondences;
    }

    cv::Mat ABt = cv::Mat::zeros(3, 3, CV_64F);

    for (int i = 0; i < number_of_correspondences; i++) {
        const double* pc = pcs + 3 * i;
        const double* pw = pws + 3 * i;

        for (int j = 0; j < 3; j++) {
            ABt.at<double>(j, 0) += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            ABt.at<double>(j, 1) += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            ABt.at<double>(j, 2) += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        }
    }

    cv::Mat W, U, Vt;
    cv::SVD::compute(ABt, W, U, Vt, cv::SVD::MODIFY_A);

    cv::Mat Rmat = U * Vt;

    // Fix reflection issue (ensure determinant is +1)
    if (cv::determinant(Rmat) < 0) {
        for (int i = 0; i < 3; ++i)
            Rmat.at<double>(2, i) *= -1;
    }

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            R[i][j] = Rmat.at<double>(i, j);

    for (int i = 0; i < 3; i++)
        t[i] = pc0[i] - R[i][0] * pw0[0] - R[i][1] * pw0[1] - R[i][2] * pw0[2];
}


void PnPsolver::print_pose(const double R[3][3], const double t[3])
{
  cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
  cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
  cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

void PnPsolver::solve_for_sign(void)
{
  if (pcs[2] < 0.0) {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
	ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

double PnPsolver::compute_R_and_t(const double * ut, const double * betas,
			     double R[3][3], double t[3])
{
  compute_ccs(betas, ut);
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void PnPsolver::find_betas_approx_1(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas)
{
    double l_6x4[6 * 4], b4[4];
    cv::Mat L_6x4(6, 4, CV_64F, l_6x4);
    cv::Mat B4(4, 1, CV_64F, b4);

    for (int i = 0; i < 6; i++) {
        L_6x4.at<double>(i, 0) = L_6x10->at<double>(i, 0);
        L_6x4.at<double>(i, 1) = L_6x10->at<double>(i, 1);
        L_6x4.at<double>(i, 2) = L_6x10->at<double>(i, 3);
        L_6x4.at<double>(i, 3) = L_6x10->at<double>(i, 6);
    }

    cv::solve(L_6x4, *Rho, B4, cv::DECOMP_SVD);

    betas[0] = std::sqrt(std::abs(b4[0]));
    betas[1] = std::sqrt(std::abs(b4[1])) * (b4[1] < 0 ? -1.0 : 1.0);
    betas[2] = b4[2];
    betas[3] = b4[3];
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void PnPsolver::find_betas_approx_2(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas)
{
    double l_6x3[6 * 3], b3[3];
    cv::Mat L_6x3(6, 3, CV_64F, l_6x3);
    cv::Mat B3(3, 1, CV_64F, b3);

    for (int i = 0; i < 6; i++) {
        L_6x3.at<double>(i, 0) = L_6x10->at<double>(i, 0);
        L_6x3.at<double>(i, 1) = L_6x10->at<double>(i, 1);
        L_6x3.at<double>(i, 2) = L_6x10->at<double>(i, 2);
    }

    cv::solve(L_6x3, *Rho, B3, cv::DECOMP_SVD);

    if (b3[0] < 0) {
        betas[0] = std::sqrt(-b3[0]);
        betas[1] = (b3[2] < 0) ? std::sqrt(-b3[2]) : 0.0;
    } else {
        betas[0] = std::sqrt(b3[0]);
        betas[1] = (b3[2] > 0) ? std::sqrt(b3[2]) : 0.0;
    }

    if (b3[1] < 0)
        betas[0] = -betas[0];

    betas[2] = 0.0;
    betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void PnPsolver::find_betas_approx_3(const cv::Mat* L_6x10, const cv::Mat* Rho, double* betas)
{
    double l_6x5[6 * 5], b5[5];
    cv::Mat L_6x5(6, 5, CV_64F, l_6x5);
    cv::Mat B5(5, 1, CV_64F, b5);

    for (int i = 0; i < 6; i++) {
        L_6x5.at<double>(i, 0) = L_6x10->at<double>(i, 0);
        L_6x5.at<double>(i, 1) = L_6x10->at<double>(i, 1);
        L_6x5.at<double>(i, 2) = L_6x10->at<double>(i, 2);
        L_6x5.at<double>(i, 3) = L_6x10->at<double>(i, 3);
        L_6x5.at<double>(i, 4) = L_6x10->at<double>(i, 4);
    }

    cv::solve(L_6x5, *Rho, B5, cv::DECOMP_SVD);

    if (b5[0] < 0) {
        betas[0] = std::sqrt(-b5[0]);
        betas[1] = (b5[2] < 0) ? std::sqrt(-b5[2]) : 0.0;
    } else {
        betas[0] = std::sqrt(b5[0]);
        betas[1] = (b5[2] > 0) ? std::sqrt(b5[2]) : 0.0;
    }

    if (b5[1] < 0)
        betas[0] = -betas[0];

    betas[2] = b5[3] / betas[0];
    betas[3] = 0.0;
}


void PnPsolver::compute_L_6x10(const double * ut, double * l_6x10)
{
  const double * v[4];

  v[0] = ut + 12 * 11;
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 *  9;
  v[3] = ut + 12 *  8;

  double dv[4][6][3];

  for(int i = 0; i < 4; i++) {
    int a = 0, b = 1;
    for(int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
	a++;
	b = a + 1;
      }
    }
  }

  for(int i = 0; i < 6; i++) {
    double * row = l_6x10 + 10 * i;

    row[0] =        dot(dv[0][i], dv[0][i]);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
    row[2] =        dot(dv[1][i], dv[1][i]);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
    row[5] =        dot(dv[2][i], dv[2][i]);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
    row[9] =        dot(dv[3][i], dv[3][i]);
  }
}

void PnPsolver::compute_rho(double * rho)
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

void PnPsolver::compute_A_and_b_gauss_newton(const double* l_6x10, const double* rho,
  double betas[4], cv::Mat* A, cv::Mat* b)
  {
    for (int i = 0; i < 6; i++) {
    const double* rowL = l_6x10 + i * 10;

    (*A).at<double>(i, 0) = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] + rowL[6] * betas[3];
    (*A).at<double>(i, 1) = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] + rowL[7] * betas[3];
    (*A).at<double>(i, 2) = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] + rowL[8] * betas[3];
    (*A).at<double>(i, 3) = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

    double value =
    rowL[0] * betas[0] * betas[0] +
    rowL[1] * betas[0] * betas[1] +
    rowL[2] * betas[1] * betas[1] +
    rowL[3] * betas[0] * betas[2] +
    rowL[4] * betas[1] * betas[2] +
    rowL[5] * betas[2] * betas[2] +
    rowL[6] * betas[0] * betas[3] +
    rowL[7] * betas[1] * betas[3] +
    rowL[8] * betas[2] * betas[3] +
    rowL[9] * betas[3] * betas[3];

    (*b).at<double>(i, 0) = rho[i] - value;
  }
}


void PnPsolver::gauss_newton(const cv::Mat* L_6x10, const cv::Mat* Rho, double betas[4])
{
    const int iterations_number = 5;

    double a[6 * 4], b[6], x[4];
    cv::Mat A(6, 4, CV_64F, a);  // Jacobian matrix
    cv::Mat B(6, 1, CV_64F, b);  // Residuals
    cv::Mat X(4, 1, CV_64F, x);  // Correction vector

    for (int k = 0; k < iterations_number; ++k) {
        compute_A_and_b_gauss_newton(
            reinterpret_cast<const double*>(L_6x10->ptr<double>()),
            reinterpret_cast<const double*>(Rho->ptr<double>()),
            betas, &A, &B
        );

        // Solve AX = B using QR decomposition
        bool success = cv::solve(A, B, X, cv::DECOMP_QR);
        if (!success) {
            std::cerr << "[PnPsolver] QR solve failed during Gauss-Newton iteration.\n";
            break;
        }

        // Update betas
        for (int i = 0; i < 4; ++i) {
            betas[i] += x[i];
        }
    }
}


// void PnPsolver::qr_solve(cv::Mat * A, cv::Mat * b, cv::Mat * X)
// {
//   static int max_nr = 0;
//   static double * A1, * A2;

//   const int nr = A->rows;
//   const int nc = A->cols;

//   if (max_nr != 0 && max_nr < nr) {
//     delete [] A1;
//     delete [] A2;
//   }
//   if (max_nr < nr) {
//     max_nr = nr;
//     A1 = new double[nr];
//     A2 = new double[nr];
//   }

//   double * pA = A->data.db, * ppAkk = pA;
//   for(int k = 0; k < nc; k++) {
//     double * ppAik = ppAkk, eta = fabs(*ppAik);
//     for(int i = k + 1; i < nr; i++) {
//       double elt = fabs(*ppAik);
//       if (eta < elt) eta = elt;
//       ppAik += nc;
//     }

//     if (eta == 0) {
//       A1[k] = A2[k] = 0.0;
//       cerr << "God damnit, A is singular, this shouldn't happen." << endl;
//       return;
//     } else {
//       double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
//       for(int i = k; i < nr; i++) {
// 	*ppAik *= inv_eta;
// 	sum += *ppAik * *ppAik;
// 	ppAik += nc;
//       }
//       double sigma = sqrt(sum);
//       if (*ppAkk < 0)
// 	sigma = -sigma;
//       *ppAkk += sigma;
//       A1[k] = sigma * *ppAkk;
//       A2[k] = -eta * sigma;
//       for(int j = k + 1; j < nc; j++) {
// 	double * ppAik = ppAkk, sum = 0;
// 	for(int i = k; i < nr; i++) {
// 	  sum += *ppAik * ppAik[j - k];
// 	  ppAik += nc;
// 	}
// 	double tau = sum / A1[k];
// 	ppAik = ppAkk;
// 	for(int i = k; i < nr; i++) {
// 	  ppAik[j - k] -= tau * *ppAik;
// 	  ppAik += nc;
// 	}
//       }
//     }
//     ppAkk += nc + 1;
//   }

//   // b <- Qt b
//   double * ppAjj = pA, * pb = b->data.db;
//   for(int j = 0; j < nc; j++) {
//     double * ppAij = ppAjj, tau = 0;
//     for(int i = j; i < nr; i++)	{
//       tau += *ppAij * pb[i];
//       ppAij += nc;
//     }
//     tau /= A1[j];
//     ppAij = ppAjj;
//     for(int i = j; i < nr; i++) {
//       pb[i] -= tau * *ppAij;
//       ppAij += nc;
//     }
//     ppAjj += nc + 1;
//   }

//   // X = R-1 b
//   double * pX = X->data.db;
//   pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
//   for(int i = nc - 2; i >= 0; i--) {
//     double * ppAij = pA + i * nc + (i + 1), sum = 0;

//     for(int j = i + 1; j < nc; j++) {
//       sum += *ppAij * pX[j];
//       ppAij++;
//     }
//     pX[i] = (pb[i] - sum) / A2[i];
//   }
// }

void PnPsolver::qr_solve(cv::Mat* A, cv::Mat* b, cv::Mat* X)
{
    if (A->rows != b->rows || A->cols != X->rows || b->cols != 1 || X->cols != 1) {
        std::cerr << "[PnPsolver] Dimension mismatch in qr_solve()." << std::endl;
        return;
    }

    // Solve A * X = b using QR decomposition
    bool success = cv::solve(*A, *b, *X, cv::DECOMP_QR);

    if (!success) {
        std::cerr << "[PnPsolver] cv::solve failed in qr_solve()." << std::endl;
    }
}




void PnPsolver::relative_error(double & rot_err, double & transl_err,
			  const double Rtrue[3][3], const double ttrue[3],
			  const double Rest[3][3],  const double test[3])
{
  double qtrue[4], qest[4];

  mat_to_quat(Rtrue, qtrue);
  mat_to_quat(Rest, qest);

  double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
			 (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
			 (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
			 (qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
			 (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
			 (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
			 (qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  rot_err = min(rot_err1, rot_err2);

  transl_err =
    sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
	 (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
	 (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
    sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
  double tr = R[0][0] + R[1][1] + R[2][2];
  double n4;

  if (tr > 0.0f) {
    q[0] = R[1][2] - R[2][1];
    q[1] = R[2][0] - R[0][2];
    q[2] = R[0][1] - R[1][0];
    q[3] = tr + 1.0f;
    n4 = q[3];
  } else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) {
    q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
    q[1] = R[1][0] + R[0][1];
    q[2] = R[2][0] + R[0][2];
    q[3] = R[1][2] - R[2][1];
    n4 = q[0];
  } else if (R[1][1] > R[2][2]) {
    q[0] = R[1][0] + R[0][1];
    q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
    q[2] = R[2][1] + R[1][2];
    q[3] = R[2][0] - R[0][2];
    n4 = q[1];
  } else {
    q[0] = R[2][0] + R[0][2];
    q[1] = R[2][1] + R[1][2];
    q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
    q[3] = R[0][1] - R[1][0];
    n4 = q[2];
  }
  double scale = 0.5f / double(sqrt(n4));

  q[0] *= scale;
  q[1] *= scale;
  q[2] *= scale;
  q[3] *= scale;
}

} //namespace ORB_SLAM

