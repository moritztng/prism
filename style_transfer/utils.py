import torch

def gram_matrix(input, feature_norm):
    b, c, h, w = input.size()
    feature_maps = input.view(b * c, h * w)
    with torch.cuda.amp.autocast(enabled=False):
        matrix = feature_maps.float() @ feature_maps.t().float()
    norm = b * c * h * w if feature_norm else h * w
    return matrix / norm

def match_color(content, style, eps=1e-5):
    content_pixels, style_pixels = content.view(3, -1), style.view(3, -1)
    mean_content, mean_style = content_pixels.mean(1), style_pixels.mean(1)
    dif_content = content_pixels - mean_content.view(3, 1)
    dif_style = style_pixels - mean_style.view(3, 1)
    eps = eps * torch.eye(3, device=content.device)
    cov_content = dif_content @ dif_content.t() / content_pixels.size(1) + eps
    cov_style = dif_style @ dif_style.t() / style_pixels.size(1) + eps
    eval_content, evec_content = torch.linalg.eig(cov_content)
    eval_content = eval_content.real.diag()
    evec_content = evec_content.real
    eval_style, evec_style = torch.linalg.eig(cov_style)
    eval_style = eval_style.real.diag()
    evec_style = evec_style.real
    cov_content_sqrt = evec_content @ eval_content.sqrt() @ evec_content.t()
    cov_style_sqrt = evec_style @ eval_style.sqrt() @ evec_style.t()
    weights = cov_style_sqrt @ cov_content_sqrt.inverse()
    bias = mean_style - weights @ mean_content
    content_pixels = weights @ content_pixels + bias.view(3, 1)
    return content_pixels.clamp(0,1).view_as(content)
