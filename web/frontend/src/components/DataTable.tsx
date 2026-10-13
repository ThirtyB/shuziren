import { renderCell } from "../lib/options";

export function DataTable({ items }: { items: Array<Record<string, unknown>> }) {
  if (!items.length) {
    return <div className="empty-table">无数据</div>;
  }
  const keys = Array.from(new Set(items.flatMap((item) => Object.keys(item)))).slice(0, 6);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {keys.map((key) => (
              <th key={key}>{key}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {items.map((item, index) => (
            <tr key={index}>
              {keys.map((key) => (
                <td key={key}>{renderCell(item[key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
