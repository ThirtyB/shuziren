import { useState } from "react";
import { Loader2, Plus, Trash2, Zap } from "lucide-react";
import type {
  PlatformSpec,
  SingleTargetMode,
  SubTaskKind,
  SubTaskSpec,
} from "../types";
import { createTestRun } from "../lib/api";
import { errorMessage } from "../lib/options";
import { SIMULATED_PLATFORMS } from "../lib/platforms";

const KIND_LABELS: Record<SubTaskKind, string> = {
  single_target: "单人定向",
  multi_user: "多人协同",
  cross_platform: "跨平台并行",
};
const MODE_LABELS: Record<SingleTargetMode, string> = {
  self_operate: "单人分发",
  attention: "数字人注意",
};

function emptySubTask(): SubTaskSpec {
  return {
    kind: "single_target",
    single_target_mode: "self_operate",
    target_user_id: "110133",
    target_old_post_id: "",
    platform_specs: [],
    with_image: false,
    event_count: 5,
    campaign_window_hours: 24,
    water_army_enabled: false,
    water_army_count: 10,
    use_llm: false,
  };
}

export type TestRunScenarioProps = {
  onStarted?: (run: { test_run_id: string }) => void;
  onError?: (message: string) => void;
  useLlm: boolean;
  setUseLlm: (value: boolean) => void;
};

/**
 * 总体测试输入面板：一个标题 + 一组子任务卡片（单人定向 / 多人协同 / 跨平台并行任意组合）。
 * 每个子任务用独立卡片呈现，可分别指定含图与特殊支持节点；事件由系统随机抽取。
 */
export function TestRunScenario(props: TestRunScenarioProps) {
  const { onStarted, onError, useLlm, setUseLlm } = props;
  const [subtasks, setSubtasks] = useState<SubTaskSpec[]>([emptySubTask()]);
  const [label, setLabel] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function updateSubtask(index: number, patch: Partial<SubTaskSpec>) {
    setSubtasks((list) => list.map((item, i) => (i === index ? { ...item, ...patch } : item)));
  }
  function addSubtask() {
    setSubtasks((list) => [...list, emptySubTask()]);
  }
  function removeSubtask(index: number) {
    setSubtasks((list) => list.filter((_, i) => i !== index));
  }

  async function handleSubmit() {
    for (const s of subtasks) {
      if (s.kind === "single_target" && !s.target_user_id) {
        onError?.("单人定向（单人分发 / 数字人注意）必须填写目标数字人。");
        return;
      }
      if (s.kind === "multi_user" && (!s.platform_specs || s.platform_specs.length === 0)) {
        onError?.("多人协同必须指定平台与人数。");
        return;
      }
      if (s.kind === "cross_platform" && (!s.platform_specs || s.platform_specs.length === 0)) {
        onError?.("跨平台（多平台并行）至少指定一个平台与人数。");
        return;
      }
    }
    setSubmitting(true);
    try {
      const request = {
        label: label.trim() || null,
        subtasks: subtasks.map((s) => ({
          kind: s.kind,
          single_target_mode: s.kind === "single_target" ? s.single_target_mode : undefined,
          target_user_id: s.kind === "single_target" ? s.target_user_id || null : null,
          target_old_post_id:
            s.kind === "single_target" && s.single_target_mode === "attention"
              ? s.target_old_post_id || null
              : null,
          platform_specs: s.platform_specs ?? [],
          with_image: !!s.with_image,
          event_count: s.event_count ?? 1,
          campaign_window_hours: s.campaign_window_hours ?? 24,
          water_army_enabled: !!s.water_army_enabled,
          water_army_count: s.water_army_enabled ? s.water_army_count ?? 0 : 0,
          use_llm: useLlm,
        })),
      };
      const created = await createTestRun(request);
      onStarted?.({ test_run_id: created.test_run_id });
    } catch (err) {
      onError?.(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="stack">
      <div className="scenario-hint">
        一个标题 + 一组子任务卡片（单人定向 / 多人协同 / 跨平台并行任意组合）；事件由系统随机抽取，含图时随机抽一张图 OCR 成事件。
        提交后异步执行，可在右侧历史查看进度与汇总计时。
      </div>

      <label className="title-input">
        测试标题
        <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder="例如 总体测试 2026-11-02" />
      </label>

      <div className="subtask-list">
        <div className="subtask-list-head">
          <h4>子任务（{subtasks.length}）</h4>
          <button type="button" className="secondary-button" onClick={addSubtask}>
            <Plus size={14} /> 添加子任务
          </button>
        </div>
        {subtasks.map((s, index) => (
          <SubTaskCard
            key={index}
            index={index}
            subtask={s}
            onChange={(patch) => updateSubtask(index, patch)}
            onRemove={() => removeSubtask(index)}
            canRemove={subtasks.length > 1}
          />
        ))}
      </div>

      <div className="scenario-hint small">
        当前顶栏 LLM 开关 = <strong>{useLlm ? "开启" : "关闭"}</strong>，本次所有子任务都按这一档运行。
        <br />
        提交后立即返回，测试在后台执行——可连续发起多次。
      </div>

      <button className="primary-button" type="button" onClick={handleSubmit} disabled={submitting}>
        {submitting ? <Loader2 className="spin" size={18} /> : <Zap size={18} />}
        开始总体测试（{subtasks.length} 个子任务）
      </button>
    </div>
  );
}

function SubTaskCard({
  index,
  subtask,
  onChange,
  onRemove,
  canRemove,
}: {
  index: number;
  subtask: SubTaskSpec;
  onChange: (patch: Partial<SubTaskSpec>) => void;
  onRemove: () => void;
  canRemove: boolean;
}) {
  const kind = subtask.kind;
  const firstSpec = subtask.platform_specs?.[0];
  return (
    <div className="subtask-card">
      <div className="subtask-card-head">
        <span className="subtask-index">#{index + 1}</span>
        <select value={kind} onChange={(e) => onChange({ kind: e.target.value as SubTaskKind })}>
          {(["single_target", "multi_user", "cross_platform"] as SubTaskKind[]).map((k) => (
            <option key={k} value={k}>
              {KIND_LABELS[k]}
            </option>
          ))}
        </select>
        <button type="button" className="run-history-delete" title="删除子任务" onClick={onRemove} disabled={!canRemove}>
          <Trash2 size={14} />
        </button>
      </div>

      <div className="form-grid">
        {kind === "single_target" && (
          <>
            <label>
              打法
              <select
                value={subtask.single_target_mode}
                onChange={(e) => onChange({ single_target_mode: e.target.value as SingleTargetMode })}
              >
                <option value="self_operate">{MODE_LABELS.self_operate}</option>
                <option value="attention">{MODE_LABELS.attention}</option>
              </select>
            </label>
            <label>
              目标数字人
              <input
                value={subtask.target_user_id ?? ""}
                onChange={(e) => onChange({ target_user_id: e.target.value })}
                placeholder="例如110133，必填"
              />
            </label>
            {subtask.single_target_mode === "attention" && (
              <label>
                原帖编号
                <input
                  value={subtask.target_old_post_id ?? ""}
                  onChange={(e) => onChange({ target_old_post_id: e.target.value })}
                  placeholder="被 @ 路人的旧帖编号"
                />
              </label>
            )}
          </>
        )}

        {kind === "multi_user" && (
          <>
            <label>
              平台
              <select
                value={firstSpec?.platform_key ?? ""}
                onChange={(e) =>
                  onChange({
                    platform_specs: [
                      { platform_key: e.target.value, requested_count: firstSpec?.requested_count ?? 5 },
                    ],
                  })
                }
              >
                <option value="">请选择平台</option>
                {SIMULATED_PLATFORMS.map((p) => (
                  <option key={p.value} value={p.value}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              人数
              <input
                type="number"
                min={1}
                max={50}
                value={firstSpec?.requested_count ?? 5}
                onChange={(e) =>
                  onChange({
                    platform_specs: [
                      {
                        platform_key: firstSpec?.platform_key ?? SIMULATED_PLATFORMS[0].value,
                        requested_count: Number(e.target.value),
                      },
                    ],
                  })
                }
              />
            </label>
          </>
        )}

        {kind === "cross_platform" && (
          <PlatformSpecEditor
            value={subtask.platform_specs ?? []}
            onChange={(specs) => onChange({ platform_specs: specs })}
          />
        )}

        <label>
          时间窗口（小时）
          <input
            type="number"
            min={1}
            max={168}
            value={subtask.campaign_window_hours ?? 24}
            onChange={(e) => onChange({ campaign_window_hours: Number(e.target.value) })}
          />
        </label>

        <label>
          随机事件条数
          <input
            type="number"
            min={1}
            max={50}
            value={subtask.event_count ?? 1}
            onChange={(e) => onChange({ event_count: Number(e.target.value) })}
          />
        </label>

        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!subtask.with_image}
            onChange={(e) => onChange({ with_image: e.target.checked })}
          />
          含图（随机抽一张图 OCR 成事件）
        </label>

        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={!!subtask.water_army_enabled}
            onChange={(e) => onChange({ water_army_enabled: e.target.checked })}
          />
          特殊支持节点参与
        </label>
        {subtask.water_army_enabled && (
          <label>
            特殊支持节点人数
            <input
              type="number"
              min={0}
              max={300}
              value={subtask.water_army_count ?? 0}
              onChange={(e) => onChange({ water_army_count: Number(e.target.value) })}
            />
          </label>
        )}
      </div>
    </div>
  );
}

function PlatformSpecEditor({
  value,
  onChange,
}: {
  value: PlatformSpec[];
  onChange: (specs: PlatformSpec[]) => void;
}) {
  function update(i: number, patch: Partial<PlatformSpec>) {
    onChange(value.map((item, idx) => (idx === i ? { ...item, ...patch } : item)));
  }
  function add() {
    onChange([...value, { platform_key: SIMULATED_PLATFORMS[0].value, requested_count: 5 }]);
  }
  function remove(i: number) {
    onChange(value.filter((_, idx) => idx !== i));
  }
  return (
    <div className="platform-spec-editor">
      <div className="platform-spec-head">
        <span>各平台与人数</span>
        <button type="button" className="secondary-button" onClick={add}>
          <Plus size={12} /> 平台
        </button>
      </div>
      {value.length === 0 && (
        <div className="empty-state small">未指定平台（跨平台模式下需至少 1 个）。</div>
      )}
      {value.map((spec, i) => (
        <div className="platform-spec-row" key={i}>
          <select value={spec.platform_key} onChange={(e) => update(i, { platform_key: e.target.value })}>
            {SIMULATED_PLATFORMS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
          <input
            type="number"
            min={1}
            max={50}
            value={spec.requested_count}
            onChange={(e) => update(i, { requested_count: Number(e.target.value) })}
            title="人数"
          />
          <button type="button" className="run-history-delete" onClick={() => remove(i)}>
            <Trash2 size={12} />
          </button>
        </div>
      ))}
    </div>
  );
}
