import { useRef, useState } from "react";
import { ImageIcon, Upload, X } from "lucide-react";

/**
 * 多模态图像事件输入。
 *
 * 后端 ``POST /api/runs/image`` 用 :envvar:`VISION_LLM_MODEL`（默认
 * ``qwen3-vl-flash``）识别图片中的新闻事件，再走与文本事件完全相同的策略生成
 * 管线。视觉模型不可用时后端会回退到 ``image_event_loader._fallback_result``，
 * 前端这边只负责：
 *
 *  1. 校验后缀（.png / .jpg / .jpeg / .webp）和大小（≤ 8 MB）；
 *  2. 展示本地预览，便于用户确认图片正确；
 *  3. 通过 :prop:`onFileChange` 把选中的 ``File`` 传给父组件（``ConsoleRoute``），
 *     由父组件的 ``submit`` 闭包在点击"运行"时组装 ``FormData`` 上传。
 *
 * 识别出的事件标题 / 摘要 / 领域 / 关键词 / 观点变体 / 置信度 / 匹配分数
 * 仍由后端在响应 trace 中给出，前端不需要再次显示。
 */

export type ImageEventFormProps = {
  onFileChange: (file: File | null) => void;
};

const SUPPORTED_SUFFIXES = new Set([".png", ".jpg", ".jpeg", ".webp"]);
const MAX_BYTES = 8 * 1024 * 1024;

function suffixOf(name: string): string {
  const index = name.lastIndexOf(".");
  return index < 0 ? "" : name.slice(index).toLowerCase();
}

export function ImageEventForm({ onFileChange }: ImageEventFormProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState("");

  function setNextFile(next: File | null) {
    setError("");
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    if (!next) {
      setFile(null);
      setPreviewUrl(null);
      onFileChange(null);
      return;
    }
    const suffix = suffixOf(next.name);
    if (!SUPPORTED_SUFFIXES.has(suffix)) {
      setError(
        `不支持的图片格式：${suffix || "(无后缀)"}（仅支持 .png / .jpg / .jpeg / .webp）`,
      );
      return;
    }
    if (next.size > MAX_BYTES) {
      setError(
        `图片过大：${(next.size / 1024 / 1024).toFixed(1)} MB（限制 8 MB）`,
      );
      return;
    }
    setFile(next);
    setPreviewUrl(URL.createObjectURL(next));
    onFileChange(next);
  }

  return (
    <div className="event-input image-event-input">
      <div className="image-event-grid">
        <div className="image-event-pick">
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(event) => setNextFile(event.target.files?.[0] ?? null)}
            hidden
          />
          <button
            type="button"
            className="secondary-button"
            onClick={() => inputRef.current?.click()}
          >
            <Upload size={18} />
            {file ? "更换图片" : "选择图片"}
          </button>
          <p className="image-event-hint">
            支持 .png / .jpg / .jpeg / .webp（≤ 8 MB），由后端视觉 LLM 识别事件标题 / 概要 / 关键词 / 观点变体后进入策略生成。
          </p>
        </div>
        <div className="image-event-preview">
          {previewUrl ? (
            <>
              <img src={previewUrl} alt="已选图片预览" />
              <div className="image-event-meta">
                <span className="image-event-name">
                  <ImageIcon size={14} /> {file?.name}
                </span>
                <span className="image-event-size">
                  {file ? `${(file.size / 1024).toFixed(1)} KB` : ""}
                </span>
                <button
                  type="button"
                  className="image-event-clear"
                  onClick={() => setNextFile(null)}
                >
                  <X size={14} /> 清除
                </button>
              </div>
            </>
          ) : (
            <div className="image-event-placeholder">
              <ImageIcon size={32} />
              <span>选择图片后在此预览</span>
            </div>
          )}
        </div>
      </div>
      {error && <div className="message error">{error}</div>}
    </div>
  );
}
