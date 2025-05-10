/**
 * 图像伪造检测系统主JS文件 - 优化版
 */

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    console.log('图像伪造检测系统已加载');
    
    // 初始化所有工具提示
    initializeTooltips();
    
    // 初始化结果页面的进度条动画（如果在结果页面）
    initializeProgressBars();

    // 启用平滑滚动
    enableSmoothScrolling();
    
    // 初始化图像预览增强功能
    initializeImagePreview();
    
    // 表单验证和提交增强
    enhanceFormSubmission();
    
    // 添加卡片悬停效果
    addCardHoverEffects();
    
    // 初始化页面动画
    initializePageAnimations();
});

/**
 * 初始化Bootstrap工具提示
 */
function initializeTooltips() {
    // 检查是否有Bootstrap的tooltip功能
    if (typeof bootstrap !== 'undefined' && typeof bootstrap.Tooltip !== 'undefined') {
        const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltips.forEach(function(tooltip) {
            new bootstrap.Tooltip(tooltip);
        });
    }
}

/**
 * 启用网站的平滑滚动
 */
function enableSmoothScrolling() {
    // 为所有内部链接添加平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 20,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // 添加返回顶部按钮功能
    const backToTopBtn = document.getElementById('backToTopBtn');
    if (backToTopBtn) {
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                backToTopBtn.classList.add('show');
            } else {
                backToTopBtn.classList.remove('show');
            }
        });
        
        backToTopBtn.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
}

/**
 * 初始化结果页面的进度条动画
 */
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    if (progressBars.length > 0) {
        // 为进度条添加初始宽度为0的样式
        progressBars.forEach(function(bar) {
            const targetWidth = bar.style.width;
            bar.style.width = '0%';
            
            // 使用Intersection Observer API监测进度条是否进入视口
            const observer = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        // 延迟执行动画以确保CSS过渡效果生效
                        setTimeout(function() {
                            bar.style.width = targetWidth;
                        }, 200);
                        observer.disconnect(); // 动画触发后停止观察
                    }
                });
            }, { threshold: 0.2 });
            
            observer.observe(bar);
        });
    }
}

/**
 * 增强图像预览功能
 */
function initializeImagePreview() {
    const fileInput = document.getElementById('file');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const uploadArea = document.getElementById('uploadArea');
    
    if (fileInput && imagePreviewContainer) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                
                // 添加加载指示器
                imagePreviewContainer.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">加载中...</span></div>';
                imagePreviewContainer.classList.remove('d-none');
                
                reader.onload = function(e) {
                    // 创建图像对象以获取尺寸
                    const img = new Image();
                    img.onload = function() {
                        // 更新预览容器内容
                        imagePreviewContainer.innerHTML = `
                            <h5>图像预览</h5>
                            <div class="image-details mb-2">
                                <span class="badge bg-info me-2">${file.type.split('/')[1].toUpperCase()}</span>
                                <span class="badge bg-secondary me-2">${Math.round(file.size / 1024)} KB</span>
                                <span class="badge bg-dark">${this.width} × ${this.height}</span>
                            </div>
                            <div class="image-preview-wrapper">
                                <img id="imagePreview" class="img-fluid img-thumbnail mt-2" style="max-height: 300px;" src="${e.target.result}" alt="图像预览">
                            </div>
                            <div class="preview-actions mt-3 d-flex justify-content-between align-items-center">
                                <button type="button" class="btn btn-sm btn-outline-secondary btn-reset-upload">
                                    <i class="bi bi-x-circle me-1"></i>重新选择
                                </button>
                                <div class="preview-info text-muted">
                                    <small><i class="bi bi-info-circle me-1"></i>可点击"上传并检测"按钮继续</small>
                                </div>
                            </div>
                        `;
                        
                        // 切换上传区域样式
                        if (uploadArea) {
                            uploadArea.classList.add('file-selected');
                        }
                        
                        // 添加淡入动画效果
                        const newImage = imagePreviewContainer.querySelector('img');
                        if (newImage) {
                            newImage.style.opacity = '0';
                            setTimeout(() => {
                                newImage.style.transition = 'opacity 0.5s ease';
                                newImage.style.opacity = '1';
                            }, 50);
                        }
                    };
                    img.src = e.target.result;
                };
                
                reader.readAsDataURL(file);
            }
        });
        
        // 允许用户点击预览区域取消选择
        imagePreviewContainer.addEventListener('click', function(e) {
            if (e.target.matches('.btn-reset-upload')) {
                // 重置文件输入
                fileInput.value = '';
                // 隐藏预览
                imagePreviewContainer.classList.add('d-none');
                // 移除上传区域选中状态
                if (uploadArea) {
                    uploadArea.classList.remove('file-selected');
                }
            }
        });
    }
}

/**
 * 增强表单提交体验
 */
function enhanceFormSubmission() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadBtnText = document.getElementById('uploadBtnText');
    const uploadSpinner = document.getElementById('uploadSpinner');
    
    if (uploadForm && uploadBtn) {
        uploadForm.addEventListener('submit', function(e) {
            // 检查是否有文件输入
            const fileInput = this.querySelector('input[type="file"]');
            if (fileInput && (!fileInput.files || fileInput.files.length === 0)) {
                e.preventDefault();
                showToast('请选择一个图像文件', 'warning');
                return false;
            }
            
            // 更新按钮状态
            if (uploadBtn && uploadBtnText && uploadSpinner) {
                uploadBtnText.textContent = '处理中...';
                uploadSpinner.classList.remove('d-none');
                uploadBtn.disabled = true;
                
                // 添加过渡动画效果
                uploadBtn.classList.add('processing');
                
                // 如果处理时间过长，更新文本提示
                setTimeout(function() {
                    if (uploadBtn.disabled) {
                        uploadBtnText.textContent = '分析中...';
                    }
                }, 3000);
            }
        });
    }
}

/**
 * 显示Toast消息通知
 * @param {string} message - 通知消息内容
 * @param {string} type - 消息类型：success, warning, danger, info
 */
function showToast(message, type = 'info') {
    // 检查是否存在toast容器，不存在则创建
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // 创建toast元素
    const toastEl = document.createElement('div');
    toastEl.className = `toast align-items-center text-white bg-${type} border-0`;
    toastEl.setAttribute('role', 'alert');
    toastEl.setAttribute('aria-live', 'assertive');
    toastEl.setAttribute('aria-atomic', 'true');
    
    // 设置toast内容
    toastEl.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="关闭"></button>
        </div>
    `;
    
    // 添加到容器并显示
    toastContainer.appendChild(toastEl);
    const toast = new bootstrap.Toast(toastEl, { autohide: true, delay: 3000 });
    toast.show();
    
    // 消失后移除DOM元素
    toastEl.addEventListener('hidden.bs.toast', function() {
        toastEl.remove();
    });
}

/**
 * 为卡片添加悬停效果
 */
function addCardHoverEffects() {
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.classList.add('card-hover');
        });
        
        card.addEventListener('mouseleave', function() {
            this.classList.remove('card-hover');
        });
    });
}

/**
 * 初始化页面动画效果
 */
function initializePageAnimations() {
    // 首先激活主内容区域 - 这是关键修复
    const mainContent = document.querySelector('.main-content');
    if (mainContent && mainContent.classList.contains('animate-ready')) {
        // 立即开始过渡到可见状态
        setTimeout(() => {
            mainContent.classList.add('animate-in');
        }, 100);
        
        // 安全措施：无论如何，500ms后强制显示内容
        setTimeout(() => {
            if (!mainContent.classList.contains('animate-in')) {
                mainContent.classList.add('animate-in');
            }
            
            // 极端情况下的备用方案
            if (getComputedStyle(mainContent).opacity < 0.5) {
                mainContent.style.opacity = '1';
                mainContent.style.transform = 'none';
            }
        }, 500);
    }
    
    // 使用Intersection Observer API为其他元素添加淡入效果
    const animateElements = document.querySelectorAll('.card, .alert, h2, h3');
    
    if (animateElements.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        
        animateElements.forEach((element, index) => {
            // 设置序号，用于交错动画
            element.style.setProperty('--child-index', index);
            
            // 如果元素尚未标记为准备动画，添加类
            if (!element.classList.contains('animate-ready')) {
                element.classList.add('animate-ready');
            }
            
            observer.observe(element);
        });
    }
    
    // 额外安全措施：确保页面加载3秒后所有动画元素都可见
    setTimeout(() => {
        document.querySelectorAll('.animate-ready:not(.animate-in)').forEach(el => {
            el.classList.add('animate-in');
        });
    }, 3000);

    // 特性项动画
    const features = document.querySelectorAll('.feature-item');
    features.forEach((feature, index) => {
        feature.style.setProperty('--child-index', index);
        feature.classList.add('animate-ready');
        
        // 设置交错动画时间
        setTimeout(() => {
            feature.classList.add('animate-in');
        }, 100 * index);
    });
    
    // 初始化所有模态框
    initializeModals();
}

/**
 * 图像文件大小验证
 * @param {HTMLInputElement} input - 文件输入元素
 * @param {number} maxSize - 最大文件大小（MB）
 * @returns {boolean} 是否通过验证
 */
function validateFileSize(input, maxSize) {
    if (!input.files || input.files.length === 0) {
        return false;
    }
    
    const fileSize = input.files[0].size / (1024 * 1024); // 转换为MB
    if (fileSize > maxSize) {
        showToast(`文件大小不能超过 ${maxSize}MB！当前文件大小: ${fileSize.toFixed(2)}MB`, 'warning');
        input.value = ''; // 清空输入
        return false;
    }
    
    return true;
}

/**
 * 图像文件类型验证
 * @param {HTMLInputElement} input - 文件输入元素
 * @param {Array} allowedTypes - 允许的文件类型数组
 * @returns {boolean} 是否通过验证
 */
function validateFileType(input, allowedTypes) {
    if (!input.files || input.files.length === 0) {
        return false;
    }
    
    const fileName = input.files[0].name;
    const fileExt = fileName.split('.').pop().toLowerCase();
    
    if (!allowedTypes.includes(fileExt)) {
        showToast(`只支持 ${allowedTypes.join(', ')} 格式的文件！`, 'warning');
        input.value = ''; // 清空输入
        return false;
    }
    
    return true;
}

// 图像上传验证
const fileInput = document.getElementById('file');
if (fileInput) {
    fileInput.addEventListener('change', function() {
        const maxSize = 16; // 16MB
        const allowedTypes = ['jpg', 'jpeg', 'png'];
        
        if (!validateFileSize(this, maxSize) || !validateFileType(this, allowedTypes)) {
            return false;
        }
    });
}

/**
 * 初始化和增强所有Bootstrap模态框
 */
function initializeModals() {
    // 检查页面上是否有模态框
    const modals = document.querySelectorAll('.modal');
    if (!modals.length) return;
    
    modals.forEach(modalEl => {
        // 确保Bootstrap加载完成
        if (typeof bootstrap === 'undefined' || typeof bootstrap.Modal === 'undefined') {
            console.warn('Bootstrap JS 未加载，模态框功能可能不正常');
            return;
        }
        
        // 为每个模态框创建Bootstrap Modal实例
        const modal = new bootstrap.Modal(modalEl);
        
        // 为所有关闭按钮添加事件处理
        const closeButtons = modalEl.querySelectorAll('[data-bs-dismiss="modal"]');
        closeButtons.forEach(button => {
            button.addEventListener('click', () => {
                modal.hide();
            });
        });
        
        // 点击背景关闭模态框（可选）
        modalEl.addEventListener('click', (e) => {
            if (e.target === modalEl) {
                modal.hide();
            }
        });
    });
    
    // 添加全局ESC键处理（可选）
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.body.classList.contains('modal-open')) {
            // 如果有打开的模态框，关闭最顶层的一个
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                const modalInstance = bootstrap.Modal.getInstance(openModal);
                if (modalInstance) modalInstance.hide();
            }
        }
    });
} 