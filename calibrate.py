import heic2png
import numpy as np
import cv2 as cv
import glob
#region Utils
OUTPUTSTEPS = True
IMGS_PATH = 'MatchBox1'

def convertimgs():
    images = glob.glob('Pics/*.HEIC')
    for name in images:
        img = heic2png.HEIC2PNG(name, quality=90)
        img.save(name.replace('.HEIC', '.png'))
        
def read_images(image_path='Calibration/*.png'):
    images = glob.glob(image_path)
    imgs = []
    for fname in images:
        img = cv.imread(fname)
        imgs.append(img)
    return imgs

def save_ply_file(points_3d, filename):
    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points_3d)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for point in points_3d:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')


def hstack_images(img1, img2):
    h1, h2 = img1.shape[0], img2.shape[0]
    if h1 < h2:
        img1 = np.pad(img1, ((0, h2 - h1), (0, 0), (0, 0)))
    elif h2 < h1:
        img2 = np.pad(img2, ((0, h1 - h2), (0, 0), (0, 0)))
    return np.hstack([img1, img2])

def draw_line(img, line, color, thickness=2):
    """
    Draw line a x + b y + c = 0 clipped to the image rectangle.

    We find intersections with all 4 borders:
      x = 0, x = w-1, y = 0, y = h-1
    and keep the points that lie inside the image.
    """
    a, b, c = line
    h, w = img.shape[:2]
    pts = []

    # Avoid division by zero issues with small numbers
    eps = 1e-8

    # Intersect with x = 0  ->  b y + c = 0
    if abs(b) > eps:
        y = -c / b
        if 0 <= y < h:
            pts.append((0, int(round(y))))

    # Intersect with x = w-1  ->  a (w-1) + b y + c = 0
    if abs(b) > eps:
        y = -(c + a * (w - 1)) / b
        if 0 <= y < h:
            pts.append((w - 1, int(round(y))))

    # Intersect with y = 0  ->  a x + c = 0
    if abs(a) > eps:
        x = -c / a
        if 0 <= x < w:
            pts.append((int(round(x)), 0))

    # Intersect with y = h-1 ->  a x + b (h-1) + c = 0
    if abs(a) > eps:
        x = -(c + b * (h - 1)) / a
        if 0 <= x < w:
            pts.append((int(round(x)), h - 1))

    # We may have more than two intersection points, pick two distinct ones
    if len(pts) >= 2:
        # Remove duplicates
        pts_unique = []
        for p in pts:
            if p not in pts_unique:
                pts_unique.append(p)

        if len(pts_unique) >= 2:
            cv.line(img, pts_unique[0], pts_unique[1], color, thickness)
    
def draw(lines1,lines2,img1_draw,img2_draw,i,j):
    draw_count = 0
    for (x1, y1), (x2, y2), l1, l2 in zip(pts1, pts2, lines1, lines2):
        color = tuple(int(255 * c) for c in np.random.rand(3))

        # draw points
        cv.circle(img1_draw, (int(round(x1)), int(round(y1))), 20, color, -1)
        cv.circle(img2_draw, (int(round(x2)), int(round(y2))), 20, color, -1)

        # draw epipolar lines
        draw_line(img1_draw, l1, color, 5)  # line in img1 for point in img2
        draw_line(img2_draw, l2, color, 5)  # line in img2 for point in img1
        draw_count+=1

        if draw_count >= 15:
            break
    cv.imwrite(f"{IMGS_PATH}_out/epipolar_lines_{i}_{j}.png", hstack_images(img1_draw, img2_draw))
#endregion

#region Calibration
def findChessBoards(chessboard_path='Calibration1/*.png'):
    """
        Find chessboard corners in the calibration images
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    img_size = None

    images = glob.glob(chessboard_path)    

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (9,6), None)
        # print (ret)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
    
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    return objpoints, imgpoints, img_size

def calibrate_camera():
    """
        Calibrate the camera using the object points and image points
    """
    obj_points, img_points, img_size = findChessBoards()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, img_size, None, None)
    
    return mtx, dist
#endregion

#region Image preparation and Feature detection
def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    
    return dst, newcameramtx

def detect_features(gray_img):
    sift = cv.SIFT_create(nfeatures=5000)
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

#endregion

#region Fundemantal matrix

def symmetric_epipolar_distance(F, pts1, pts2):
    homog1 = np.hstack([pts1, np.ones((len(pts1), 1))])
    homog2 = np.hstack([pts2, np.ones((len(pts2), 1))])
    l2 = (F @ homog1.T).T
    l1 = (F.T @ homog2.T).T

    num1 = np.abs(np.sum(l2 * homog2, axis=1))
    num2 = np.abs(np.sum(l1 * homog1, axis=1))

    denom1 = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2)
    denom2 = np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)

    # divide by zero error
    d1 = np.where(denom1 > 1e-8, num1 / denom1, np.inf)
    d2 = np.where(denom2 > 1e-8, num2 / denom2, np.inf)
    return 0.5 * (d1 + d2)

def normalize_points(points):
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / mean_dist
    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ])
    homog = np.hstack([points, np.ones((points.shape[0], 1))])
    normalized = (T @ homog.T).T
    return normalized[:, :2], T


def build_design_matrix(x1, x2):
    # x1: (N,2) from image 1  -> (x1,y1)
    # x2: (N,2) from image 2  -> (x2,y2)
    X1 = x1[:, 0]
    Y1 = x1[:, 1]
    X2 = x2[:, 0]
    Y2 = x2[:, 1]

    return np.column_stack([
        X1 * X2,     # f11 term
        Y1 * X2,     # f12 term
        X2,          # f13 term
        X1 * Y2,     # f21 term
        Y1 * Y2,     # f22 term
        Y2,          # f23 term
        X1,          # f31 term
        Y1,          # f32 term
        np.ones(len(x1))  # f33 term
    ])

def eight_point_F(x1, x2):
    A = build_design_matrix(x1, x2)
    _, _, vh = np.linalg.svd(A)
    F = vh[-1].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    return U @ np.diag(S) @ Vt

def ransac_fundamental(pts1, pts2, num_iters=3000, sample_size=8, threshold=1.5):
    N = pts1.shape[0]
    best_inliers = None
    best_F = None
    best_count = 0
    rng = np.random.default_rng(42)

    for _ in range(num_iters):
        idx = rng.choice(N, size=sample_size, replace=False)

        # Normalize only the sample — not all N points
        n1, T1 = normalize_points(pts1[idx])
        n2, T2 = normalize_points(pts2[idx])

        F_norm = eight_point_F(n1, n2)
        F_cand = T2.T @ F_norm @ T1
        if np.abs(F_cand[2, 2]) > 1e-8:
            F_cand /= F_cand[2, 2]

        residuals = symmetric_epipolar_distance(F_cand, pts1, pts2)
        inliers = residuals < threshold
        count = inliers.sum()

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_F = F_cand

    # Refit F
    if best_inliers is not None and best_inliers.sum() >= 8:
        n1, T1 = normalize_points(pts1[best_inliers])
        n2, T2 = normalize_points(pts2[best_inliers])
        F_refitted = eight_point_F(n1, n2)
        best_F = T2.T @ F_refitted @ T1
        if np.abs(best_F[2, 2]) > 1e-8:
            best_F /= best_F[2, 2]
        # Recount inliers with the refitted F
        residuals = symmetric_epipolar_distance(best_F, pts1, pts2)
        best_inliers = residuals < threshold

    return best_F, best_inliers


def match_brute_force(desc1, desc2):
    bf = cv.BFMatcher(cv.NORM_L2)
    
    matches_12 = bf.knnMatch(desc1, desc2, k=2)
    matches_21 = bf.knnMatch(desc2, desc1, k=2)
    good_12 = {}
    for m, n in matches_12:
        if m.distance < 0.75 * n.distance:
            good_12[m.queryIdx] = m

    # Keep the ebst matches
    good = []
    for m, n in matches_21:
        if m.distance < 0.75 * n.distance:
            if m.trainIdx in good_12:
                fwd = good_12[m.trainIdx]
                if fwd.trainIdx == m.queryIdx:
                    good.append(fwd)

    return sorted(good, key=lambda x: x.distance)

def calculate_all_fundamental_matrices(kps_descs):
    
    results = {}
    n = len(kps_descs)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        kp1, desc1 = kps_descs[i]
        kp2, desc2 = kps_descs[j]
        kp1 = np.array([kp.pt for kp in kp1])
        kp2 = np.array([kp.pt for kp in kp2])
        
        if desc1 is None or desc2 is None:
            continue
        matches = match_brute_force(desc1, desc2)

        pts1 = np.float32([kp1[m.queryIdx] for m in matches])
        pts2 = np.float32([kp2[m.trainIdx] for m in matches])
        
        q_idx = np.array([m.queryIdx for m in matches])
        t_idx = np.array([m.trainIdx for m in matches])
        
        if len(pts1) < 8 or len(pts2) < 8:
            continue
        
        norm1, T1 = normalize_points(pts1)
        norm2, T2 = normalize_points(pts2)
        
        F, inlier_mask = ransac_fundamental(pts1, pts2,
                                    num_iters=3000,
                                    sample_size=8,
                                    threshold=10.0)
        if F is None or inlier_mask is None or inlier_mask.sum() < 8:
            continue

        results[(i, j)] = F, inlier_mask, pts1[inlier_mask], pts2[inlier_mask], q_idx[inlier_mask], t_idx[inlier_mask]
        
        # print(len(pts1[inlier_mask]))   
    return results

#endregion

#region Essential, Triangulation

def enforce_essential_constraints(E):
    U, S, Vt = np.linalg.svd(E)
    avg = 0.5 * (S[0] + S[1])
    S_new = np.array([avg, avg, 0.0])
    return U @ np.diag(S_new) @ Vt

def decompose_essential_matrix(F, mtx, pts1, pts2):
    E = mtx.T @ F @ mtx
    E = enforce_essential_constraints(E)

    _, R, t, pose_mask = cv.recoverPose(E, pts1, pts2, mtx)
    return E, R, t, pose_mask.ravel().astype(bool)

def triangulate_midpoint(P1, P2, pts1, pts2):
    M1, M2 = P1[:, :3], P2[:, :3]

    # Camera centres:  C = -M^{-1} p4
    C1 = -np.linalg.solve(M1, P1[:, 3])
    C2 = -np.linalg.solve(M2, P2[:, 3])
    w = C1 - C2                          # baseline vector

    points_3d = []
    for p1, p2 in zip(pts1, pts2):
        # Ray directions in world space: d = M^{-1} x̃
        d1 = np.linalg.solve(M1, [p1[0], p1[1], 1.0])
        d2 = np.linalg.solve(M2, [p2[0], p2[1], 1.0])
        d1 /= np.linalg.norm(d1)
        d2 /= np.linalg.norm(d2)

        a = np.dot(d1, d1)   # 1.0 after normalisation
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)   # 1.0 after normalisation
        d = np.dot(d1, w)
        e = np.dot(d2, w)

        denom = a * c - b * b
        if abs(denom) < 1e-10:   # parallel rays — degenerate, skip
            continue

        t = (b * e - c * d) / denom
        s = (a * e - b * d) / denom

        pt1 = C1 + t * d1
        pt2 = C2 + s * d2
        points_3d.append((pt1 + pt2) * 0.5)
        
    if not points_3d:
        return np.empty((0, 3))
    return np.array(points_3d)

#endregion

#region Pointcloud registration

def compute_similarity_transform(src, dst):
    n = len(src)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c  = src - mu_src
    dst_c  = dst - mu_dst

    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    if var_src < 1e-8:
        return None  # degenerate — correspondences are co-located

    H = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    R = U @ D @ Vt
    s = float(np.sum(S * np.diag(D)) / var_src)

    if not (0.01 < s < 100):       # keep sane scales
        return None

    t = mu_dst - s * (R @ mu_src)
    return s, R, t

def apply_transform(pts, s, R, t):
    return (s * (R @ pts.T)).T + t

def find_cloud_correspondences(pair_a, pair_b, kp_idx_store,
                                cloud_kp_q, cloud_kp_t, clouds):
    i, j = pair_a
    k, l = pair_b

    shared = {i, j} & {k, l}
    if not shared:
        return None, None
    shared_img = shared.pop()

    # All ransack inliers
    kp_a_all = kp_idx_store[pair_a][0 if shared_img == i else 1]
    kp_b_all = kp_idx_store[pair_b][0 if shared_img == k else 1]
    shared_vals, ia, ib = np.intersect1d(kp_a_all, kp_b_all, return_indices=True)

    # triangulated points
    lookup_a = cloud_kp_q[pair_a] if shared_img == i else cloud_kp_t[pair_a]
    lookup_b = cloud_kp_q[pair_b] if shared_img == k else cloud_kp_t[pair_b]

    corr_a, corr_b = [], []
    for kp_val in shared_vals:
        if kp_val not in lookup_a or kp_val not in lookup_b:
            continue
        ca = clouds[pair_a][lookup_a[kp_val]]
        cb = clouds[pair_b][lookup_b[kp_val]]
        if not (np.any(np.isnan(ca)) or np.any(np.isnan(cb))):
            corr_a.append(ca)
            corr_b.append(cb)

    if len(corr_a) < 3:
        return None, None

    return np.array(corr_a), np.array(corr_b)


def register_point_clouds(clouds, kp_idx_store):
    pairs = list(clouds.keys())
    if not pairs:
        return np.empty((0, 3))
    
    #find the pairs that has the most points in kp_idx_store
    pairs = sorted(pairs, key=lambda p: len(kp_idx_store[p]), reverse=True)
    ref = pairs[0]

    transforms = {ref: (1.0, np.eye(3), np.zeros(3))}
    registered = {ref}
    queue      = [ref]

    while queue:
        anchor = queue.pop(0)
        s_a, R_a, t_a = transforms[anchor]

        for cand in pairs:
            if cand in registered:
                continue

            corr_anchor, corr_cand = find_cloud_correspondences(
                anchor, cand, kp_idx_store, cloud_kp_q, cloud_kp_t, clouds
            )
            if corr_anchor is None:
                continue
            # bring points to world coordinates
            corr_anchor_world = apply_transform(corr_anchor, s_a, R_a, t_a)

            result = compute_similarity_transform(corr_cand, corr_anchor_world)
            if result is None:
                continue
            s_c, R_c, t_c = result
            transforms[cand] = (s_c, R_c, t_c)
            registered.add(cand)
            queue.append(cand)

    if len(registered) < len(pairs):
        unregistered = set(map(str, pairs)) - set(map(str, registered))
        print(f"{len(pairs) - len(registered)} cloud(s) could not be concatenated :{unregistered}")

    # Apply transforms and concatenate
    merged = []
    for pair in pairs:
        if pair not in transforms:
            continue
        pts   = clouds[pair]
        valid = ~np.any(np.isnan(pts), axis=1)
        pts   = apply_transform(pts[valid], *transforms[pair])
        merged.append(pts)

    return np.concatenate(merged, axis=0)

#endRegion


#region Main
if __name__ == "__main__":

    #Calibration
    mtx, dist = calibrate_camera()
    
    #Image undistortion
    imgs = read_images(f"{IMGS_PATH}/*.png")
    
    undist_data = [undistort_image(img, mtx, dist) for img in imgs]
    imgs = [data[0] for data in undist_data]
    
    mtx = undist_data[0][1]  
    imgs_gray = [cv.cvtColor(img, cv.COLOR_BGR2GRAY) for img in imgs]

    #Detect features
    kps_descs = [detect_features(gray) for gray in imgs_gray]
    if OUTPUTSTEPS:
        for i, (kp, desc) in enumerate(kps_descs):
            img_kp = cv.drawKeypoints(imgs[i], kp, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv.imwrite(f"{IMGS_PATH}_out/keypoints_{i}.png", img_kp)
            
    #Calculate fundamental matrixes
    fundamental_matrices = calculate_all_fundamental_matrices(kps_descs)
    
    #Get 3D points 
    clouds      = {}
    cloud_kp_q  = {} 
    cloud_kp_t  = {}
    kp_idx_store = {} 
    for (i, j), (F, mask, pts1, pts2, q_idx, t_idx) in fundamental_matrices.items():
        if OUTPUTSTEPS:
            pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # (N,3)
            pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])  # (N,3)

            lines2 = (F @ pts1_h.T).T 
            lines1 = (F.T @ pts2_h.T).T 
            
            img1_draw = imgs[i].copy()
            img2_draw = imgs[j].copy()
            draw(lines1, lines2, img1_draw, img2_draw, i, j)
        # Decompose essential matrix and triangulate
        E, R, t_vec, pose_mask = decompose_essential_matrix(F, mtx, pts1, pts2)

        P1 = mtx @ np.hstack([np.eye(3),    np.zeros((3, 1))])
        P2 = mtx @ np.hstack([R, t_vec.reshape(3, 1)])

        pts1_ch, pts2_ch = pts1[pose_mask], pts2[pose_mask]
        q_ch,    t_ch    = q_idx[pose_mask], t_idx[pose_mask]

        clouds[(i, j)]    = triangulate_midpoint(P1, P2, pts1_ch, pts2_ch)
        
        # store for registration
        cloud_kp_q[(i,j)] = {kp: ci for ci, kp in enumerate(q_ch)}  
        cloud_kp_t[(i,j)] = {kp: ci for ci, kp in enumerate(t_ch)}
        kp_idx_store[(i,j)] = (q_idx, t_idx)    
          
    merged = register_point_clouds(clouds, kp_idx_store)
    save_ply_file(merged, f"{IMGS_PATH}_out/points.ply")
    print(f"Saved {len(merged)} points.")
